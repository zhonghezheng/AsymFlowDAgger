"""How much of the logged cfg_gap can training actually remove?

The band's CFG-gap loss is a plain mean of || (v_c - v_u) - (v*_c - v*_u) ||^2 with both
branches from ONE batched train-mode forward over [x; x] (GaussianFlowOnPolicy.
_band_point_losses; GaussianFlowDagger's buffered gap likewise). Dropout
(proj_dropout=0.2, middle half of the blocks) draws an INDEPENDENT mask for each half, and
v* = (x_t - x0_hat)/sigma averages a RANDOM null bank (per-trajectory draw, ESS ~1-5).
So in expectation over masks m1, m2 and bank draws

    E[cfg_gap] = || E[v_c - v_u] - E[D*] ||^2       <- what training can remove
               + Var_m(v_c) + Var_m(v_u)            <- dropout floor (independent masks)
               + Var_bank(D*)                       <- target floor

where D* = v*_c - v*_u. Measured at the band's own rollout states (the loaded model's
Heun rollout, every landing state plus the t=1 start), per sigma:

  cfg_gap      the loss as trained: K independent-mask train forwards vs bank draw A
  drop_c/u     per-element dropout variance of each branch (K masks)
  drop_shared  Var_m(v_c - v_u) when both branches share ONE mask (CUDA RNG state reset
               between two forwards) -- the dropout floor a common-mask gap would keep
  tgt_var      Var_bank(D*) = ||D*_A - D*_B||^2 / 2 over two independent bank draws
  bias         || mean_k(v_c - v_u) - D*_A ||^2 (includes tgt_var: D*_A is one draw)
  gap_eval     the same gap with dropout OFF (eval mode, what inference runs)
  |D*|^2, |dv_eval|^2   target and model CFG signal sizes; cos(dv_eval, D*_A) per row

  floor_ind    drop_c + drop_u + tgt_var : the loss's floor with independent masks
  floor_sh     drop_shared + tgt_var     : with a shared mask
  floor_eval   tgt_var                   : with dropout off

All values are plain per-element means, the scale cfg_gap is logged on. Run on ONE GPU
(not the login node), e.g. pretrained vs the cmff w=0 / w=100 runs:

  LAKON_U8_CACHE=data/cached_latents/train_u8_256 python tools/cfg_gap_floor.py
  LAKON_U8_CACHE=... python tools/cfg_gap_floor.py \
      --ckpt checkpoints/asymflow_h_16_r8_imagenet_dagger_bankfull_cmff0_f50_4gpus/latest.pth
  LAKON_U8_CACHE=... python tools/cfg_gap_floor.py \
      --ckpt checkpoints/asymflow_h_16_r8_imagenet_dagger_bankfull_cmff100_f50_4gpus/latest.pth

Bank IO: two draws of n * (~1300 x 2 cond + 2048 null) images (~110k at n=16).
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/scratch/gpfs/AM43/zz8976/LakonLab')
os.chdir('/scratch/gpfs/AM43/zz8976/LakonLab')
from mmcv import Config  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402

from lakonlab.models import build_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--config', default='configs/asymflow/asymflow_h_16_r8_imagenet_dagger_bankfull_cfg_4gpus.py')
ap.add_argument('--ckpt', default=None, help='None -> the released pretrained weights')
ap.add_argument('--ema', action='store_true', help='measure the EMA weights (what eval samples)')
ap.add_argument('--n', type=int, default=16, help='trajectories (rows) per band state')
ap.add_argument('--k', type=int, default=8, help='dropout mask draws per state')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--tag', default=None)
args = ap.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
dev = torch.device('cuda')

cfg = Config.fromfile(args.config)
# eager, no backward here: compile and per-block checkpointing only cost time
cfg.model.diffusion.denoising.compile_forward = False
cfg.model.diffusion.denoising.checkpointing = False
model = build_model(cfg.model)
if args.ckpt:
    load_checkpoint(model, args.ckpt, map_location='cpu', strict=False)
if args.ema:
    model.diffusion.load_state_dict(model.diffusion_ema.state_dict())
model = model.to(dev).train()
for p in model.parameters():
    p.requires_grad_(False)
d = model.diffusion
expert = d._dagger_expert
assert expert is not None and expert.ready
d._band_t = {}
t_split = d.mmd_t_split
null = d.null_label
print(f'ckpt={args.ckpt or "pretrained"} ema={args.ema} t_split={t_split} n={args.n} '
      f'k={args.k} proj_dropout={cfg.model.diffusion.denoising.get("proj_dropout")}', flush=True)

# the band's classes and rollout, as GaussianFlowOnPolicy draws them ('prior')
labels = expert.sample_labels(args.n, dev)
x_ref = torch.zeros(args.n, 3, 256, 256, device=dev)   # shape only: the rollout draws its noise
with torch.no_grad():
    states = d._band_rollout(x_ref, labels, t_split, include_start=True)
print('states: ' + ' '.join(f'{s:.3f}' for s, _ in states), flush=True)

t0 = time.time()
banks_a = expert.build_banks(labels, model._expert_encode_fn, d.feat_fn, dev)
banks_b = expert.build_banks(labels, model._expert_encode_fn, d.feat_fn, dev)
print(f'two bank draws in {time.time() - t0:.0f}s', flush=True)


def velocity(x, t, lab):
    """The band's velocity, exactly as _band_point_losses forms it."""
    _, _, cc = d.get_clamp_coef(t=t, x_t=x)
    return d.pred(x, t, class_labels=lab) * cc


def pair(x, t, lab):
    """[x; x] with [labels; null] in ONE forward -- the training forward."""
    out = velocity(torch.cat([x, x]), torch.cat([t, t]),
                   torch.cat([lab, torch.full_like(lab, null)]))
    return out.chunk(2, dim=0)


def target_gap(x, sig, banks):
    x0_c, x0_u = d._expert_velocities(x, sig, labels, banks)
    return (d.expert_target(x, sig, x0_c, mode='full')
            - d.expert_target(x, sig, x0_u, mode='full'))


def msq(z):
    return float(z.float().pow(2).mean())


def subfrac(z):
    z = z.float().reshape(-1, *z.shape[-3:])   # [K, n, C, H, W] -> [K*n, C, H, W]
    return float(d.project_fn(z).pow(2).sum() / z.pow(2).sum().clamp_min(1e-30))


rows = []
with torch.no_grad():
    for sigma, x in states:
        x = x.float()
        sig = x.new_full((args.n, ), sigma)
        t = sig * d.num_timesteps
        dstar_a = target_gap(x, sig, banks_a)
        dstar_b = target_gap(x, sig, banks_b)

        # independent masks: the forward the loss actually takes
        vc, vu = [], []
        for _ in range(args.k):
            c, u = pair(x, t, labels)
            vc.append(c)
            vu.append(u)
        vc, vu = torch.stack(vc), torch.stack(vu)
        dv = vc - vu

        # one shared mask per draw: reset the CUDA RNG between the two branches
        dv_sh, repro = [], None
        for k in range(args.k):
            st = torch.cuda.get_rng_state(dev)
            c = velocity(x, t, labels)
            if k == 0:   # same state, same label -> must be bit-identical
                torch.cuda.set_rng_state(st, dev)
                repro = float((velocity(x, t, labels) - c).abs().max())
            torch.cuda.set_rng_state(st, dev)
            u = velocity(x, t, torch.full_like(labels, null))
            dv_sh.append(c - u)
        dv_sh = torch.stack(dv_sh)

        d.denoising.eval()
        c_e, u_e = pair(x, t, labels)
        d.denoising.train()
        dv_e = c_e - u_e

        drop_c = float(vc.var(0).mean())
        drop_u = float(vu.var(0).mean())
        drop_sh = float(dv_sh.var(0).mean())
        tgt_var = msq(dstar_a - dstar_b) / 2
        cos = torch.nn.functional.cosine_similarity(
            dv_e.flatten(1), dstar_a.flatten(1), dim=1).mean()
        rows.append(dict(
            sigma=sigma,
            cfg_gap=float((dv - dstar_a).pow(2).mean()),
            cfg_gap_shared=float((dv_sh - dstar_a).pow(2).mean()),
            gap_eval=msq(dv_e - dstar_a),
            bias=msq(dv.mean(0) - dstar_a),
            drop_c=drop_c, drop_u=drop_u, drop_shared=drop_sh, tgt_var=tgt_var,
            floor_ind=drop_c + drop_u + tgt_var,
            floor_sh=drop_sh + tgt_var,
            floor_eval=tgt_var,
            dstar_sq=msq(dstar_a), dv_eval_sq=msq(dv_e), cos_dv_dstar=float(cos),
            drop_subfrac=subfrac(vc - vc.mean(0)),
            tgt_subfrac=subfrac(dstar_a - dstar_b),
            gap_subfrac=subfrac(dv - dstar_a),
            mask_repro_maxdiff=repro))
        print(f'  sigma {sigma:.3f} done', flush=True)

land = [r for r in rows if r['sigma'] < 0.999]
pooled = {k: float(np.mean([r[k] for r in land])) for k in land[0] if k != 'sigma'}
pooled['sigma'] = -1.0   # mean over the landing states (what the default arms score)

cols = ['cfg_gap', 'drop_c', 'drop_u', 'drop_shared', 'tgt_var', 'bias', 'gap_eval',
        'floor_ind', 'floor_sh', 'dstar_sq', 'dv_eval_sq', 'cos_dv_dstar']
print('\n' + f'{"sigma":>7} ' + ' '.join(f'{c:>12}' for c in cols))
for r in rows + [pooled]:
    lab = 'landing' if r['sigma'] < 0 else f'{r["sigma"]:.3f}'
    print(f'{lab:>7} ' + ' '.join(f'{r[c]:12.4e}' for c in cols))
p = pooled
print(f'\nlanding states, as fractions of cfg_gap ({p["cfg_gap"]:.4e}):'
      f'\n  dropout floor, independent masks  {(p["drop_c"] + p["drop_u"]) / p["cfg_gap"]:6.1%}'
      f'\n  dropout floor, shared mask        {p["drop_shared"] / p["cfg_gap"]:6.1%}'
      f'\n  target (bank) floor               {p["tgt_var"] / p["cfg_gap"]:6.1%}'
      f'\n  removable at best (1 - floor_ind) {1 - p["floor_ind"] / p["cfg_gap"]:6.1%}'
      f'\n  subspace share: dropout {p["drop_subfrac"]:.3f}  target noise {p["tgt_subfrac"]:.3f}'
      f'  gap residual {p["gap_subfrac"]:.3f}'
      f'\n  shared-mask RNG reproducibility (must be 0): {max(r["mask_repro_maxdiff"] for r in rows):.2e}')

tag = args.tag or (os.path.basename(os.path.dirname(args.ckpt)) if args.ckpt else 'pretrained')
tag += '_ema' if args.ema else ''
out = f'direct_outputs/cfg_gap_floor_{tag}.json'
json.dump(dict(args=vars(args), rows=rows, pooled=pooled), open(out, 'w'), indent=1)
print(f'\nwrote {out}')
