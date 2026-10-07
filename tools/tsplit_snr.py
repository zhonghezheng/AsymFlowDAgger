"""How far down in sigma must the MMD band reach before the statistic clears its
own noise floor?

At every band sigma this forms exactly the statistic the training loss forms --
m=256 pooled rollout rows (64 per rank x 4) vs n=2048 pooled class-matched target
rows (512 per rank) -- and compares it against a NULL built at identical shapes
from two INDEPENDENT draws of the target distribution.  The unbiased estimator has
E[MMD^2]=0 under the null, so sd(null) is the floor and z = obs/sd(null) is the
usable signal.  Run under torchrun so the pooling is the real one.
"""
import os, sys, json, time
import numpy as np, torch
import torch.distributed as dist

sys.path.insert(0, '/scratch/gpfs/AM43/zz8976/LakonLab')
os.chdir('/scratch/gpfs/AM43/zz8976/LakonLab')

from mmcv import Config
from mmcv.runner import load_checkpoint
from lakonlab.models import build_model
from lakonlab.models.diffusions.gaussian_flow_mmd import mmd2_rbf

CKPT    = 'checkpoints/asymflow_h_16_r8_imagenet_regft_4gpus/latest.pth'
M_LOCAL = 64      # mmd_batch
N_LOCAL = 512     # mmd_target_n / world_size
REPS    = 3
T_LOW   = 0.40    # roll this far down so every candidate t_split is covered
OUT     = 'direct_outputs/tsplit_snr.json'

dist.init_process_group('nccl')
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank % torch.cuda.device_count())
dev = torch.device('cuda')


def log(*a):
    if rank == 0:
        print(*a, flush=True)


cfg = Config.fromfile('configs/asymflow/asymflow_h_16_r8_imagenet_regft_mmd_sweep_4gpus.py')
cfg.model.diffusion.mmd_target_n = N_LOCAL * world
model = build_model(cfg.model)
load_checkpoint(model, CKPT, map_location='cpu')
model = model.to(dev).eval()
for p in model.parameters():
    p.requires_grad_(False)

d = model.diffusion
d._encode_fn = model._expert_encode_fn
d.mmd_t_hi = None
d.mmd_classes_per_batch = None
d.mmd_step_checkpoint = False
d.mmd_gather = True

# rank-dependent labels so the pooled 256 look like a real sharded batch
g = torch.Generator(device='cuda').manual_seed(1234 + rank)
labels = torch.randint(0, int(d.denoising.num_classes), (M_LOCAL,), generator=g, device=dev)

probe = model._expert_encode_fn(torch.zeros(1, 3, 256, 256, device=dev))
x_ref = torch.zeros((M_LOCAL,) + probe.shape[1:], device=dev)
log(f'diffusion input space {tuple(probe.shape[1:])} -> flat '
    f'{int(np.prod(probe.shape[1:]))}; pooled m={M_LOCAL*world} n={N_LOCAL*world}')


def target_feats(per_row, sigma, space):
    x0 = d._draw_target_latents([int(c) for c in labels], per_row, dev)
    x = x0 * (1.0 - sigma) + torch.randn_like(x0) * sigma
    return d._mmd_feats(x, space)


rows, t0 = {}, time.time()
for rep in range(REPS):
    with torch.no_grad():
        states = d._band_rollout(x_ref, labels, T_LOW)
    log(f'[rep {rep}] {len(states)} band states, sigma '
        f'{states[0][0]:.4f} .. {states[-1][0]:.4f}  ({time.time()-t0:.0f}s)')
    for si, (sigma, x_roll) in enumerate(states):
        if si % 2:                      # every other state keeps the cost sane
            continue
        with torch.no_grad():
            roll = {sp: d._gather_feats(d._mmd_feats(x_roll, sp))[0]
                    for sp in ('subspace', 'raw')}
            big = {sp: d._gather_feats(target_feats(N_LOCAL // M_LOCAL, sigma, sp))[0]
                   for sp in ('subspace', 'raw')}
            sml = {sp: d._gather_feats(target_feats(1, sigma, sp))[0]
                   for sp in ('subspace', 'raw')}
            for sp in ('subspace', 'raw'):
                obs = mmd2_rbf(roll[sp], big[sp], bandwidths=d.mmd_bandwidths,
                               unbiased=d.mmd_unbiased)
                null = mmd2_rbf(sml[sp], big[sp], bandwidths=d.mmd_bandwidths,
                                unbiased=d.mmd_unbiased)
                rows.setdefault((round(sigma, 5), sp), []).append((float(obs), float(null)))
        del roll, big, sml
        torch.cuda.empty_cache()
        if si % 6 == 0:
            log(f'    sigma {sigma:.4f} done ({time.time()-t0:.0f}s)')
    log(f'[rep {rep}] complete ({time.time()-t0:.0f}s)')

if rank == 0:
    out = []
    for (sigma, sp), v in sorted(rows.items(), reverse=True):
        o = np.array([a for a, _ in v]); n = np.array([b for _, b in v])
        out.append(dict(sigma=sigma, space=sp, obs=o.mean(), obs_sd=o.std(ddof=1),
                        null=n.mean(), null_sd=n.std(ddof=1)))
    json.dump(out, open(OUT, 'w'), indent=1, default=float)

    for sp in ('subspace', 'raw'):
        print(f'\n=== {sp} : per-sigma ===')
        print(f'{"sigma":>7} {"obs MMD2":>12} {"null mean":>12} {"null sd":>10} {"z":>8}')
        for r in out:
            if r['space'] != sp:
                continue
            z = r['obs'] / r['null_sd'] if r['null_sd'] > 0 else float('nan')
            print(f'{r["sigma"]:7.4f} {r["obs"]:12.3e} {r["null"]:12.3e} '
                  f'{r["null_sd"]:10.2e} {z:8.1f}')

    # what a given t_split would buy: the loss is the MEAN over states with sigma>=t
    print(f'\n=== band-averaged signal by t_split ===')
    print(f'{"t_split":>8} {"#states":>8} {"sub mean z":>12} {"raw mean z":>12}')
    sigs = sorted({r['sigma'] for r in out}, reverse=True)
    for t in (0.95, 0.925, 0.90, 0.875, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.40):
        sel = [s for s in sigs if s >= t]
        if not sel:
            continue
        zs = {}
        for sp in ('subspace', 'raw'):
            z = [r['obs'] / r['null_sd'] for r in out
                 if r['space'] == sp and r['sigma'] in sel and r['null_sd'] > 0]
            zs[sp] = float(np.mean(z)) if z else float('nan')
        print(f'{t:8.3f} {len(sel):8d} {zs["subspace"]:12.1f} {zs["raw"]:12.1f}')
    print(f'\n(current mmd_t_split = 0.875)')

dist.barrier()
dist.destroy_process_group()
