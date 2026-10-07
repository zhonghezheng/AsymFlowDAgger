"""Does raising mmd_batch actually buy MMD signal?

tools/tsplit_snr.py showed the statistic sits at its noise floor at the current
pooled m=256, and argued the binding constraint is m rather than the band
location: for P==Q the unbiased MMD^2 variance keeps a 1/(m(m-1)) term from the
ROLLOUT side that the enlarged target side cannot compensate for.  This sweeps
m directly -- 64/128/256 rows per rank, i.e. pooled 256/512/1024 -- with the
target side held at n=2048 pooled, and reports both the null sd (does it fall
like 1/m?) and the resulting z.
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

CKPT     = 'checkpoints/asymflow_h_16_r8_imagenet_regft_4gpus/latest.pth'
M_LOCALS = (64, 128, 256)     # pooled 256 / 512 / 1024
N_LOCAL  = 512                # pooled 2048, held fixed
REPS     = 5
T_LOW    = 0.40
EVERY    = 3                  # score every 3rd band state
OUT      = 'direct_outputs/tsplit_snr_m.json'

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

probe = model._expert_encode_fn(torch.zeros(1, 3, 256, 256, device=dev))
log(f'input space {tuple(probe.shape[1:])}; n pooled = {N_LOCAL*world}, reps = {REPS}')

rows, t0 = {}, time.time()
for m_local in M_LOCALS:
    g = torch.Generator(device='cuda').manual_seed(1234 + rank)
    labels = torch.randint(0, int(d.denoising.num_classes), (m_local,),
                           generator=g, device=dev)
    x_ref = torch.zeros((m_local,) + probe.shape[1:], device=dev)
    per_row = max(1, N_LOCAL // m_local)

    def target_feats(pr, sigma, space):
        x0 = d._draw_target_latents([int(c) for c in labels], pr, dev)
        x = x0 * (1.0 - sigma) + torch.randn_like(x0) * sigma
        return d._mmd_feats(x, space)

    log(f'\n--- m pooled = {m_local*world} (per rank {m_local}), '
        f'per_row = {per_row} -> target {per_row*m_local*world} pooled')
    for rep in range(REPS):
        with torch.no_grad():
            states = d._band_rollout(x_ref, labels, T_LOW)
        for si, (sigma, x_roll) in enumerate(states):
            if si % EVERY:
                continue
            with torch.no_grad():
                roll = {sp: d._gather_feats(d._mmd_feats(x_roll, sp))[0]
                        for sp in ('subspace', 'raw')}
                big = {sp: d._gather_feats(target_feats(per_row, sigma, sp))[0]
                       for sp in ('subspace', 'raw')}
                sml = {sp: d._gather_feats(target_feats(1, sigma, sp))[0]
                       for sp in ('subspace', 'raw')}
                for sp in ('subspace', 'raw'):
                    obs = mmd2_rbf(roll[sp], big[sp], bandwidths=d.mmd_bandwidths,
                                   unbiased=d.mmd_unbiased)
                    null = mmd2_rbf(sml[sp], big[sp], bandwidths=d.mmd_bandwidths,
                                    unbiased=d.mmd_unbiased)
                    rows.setdefault((m_local * world, round(sigma, 5), sp), []).append(
                        (float(obs), float(null)))
            del roll, big, sml
            torch.cuda.empty_cache()
        log(f'    rep {rep} done ({time.time()-t0:.0f}s)')

if rank == 0:
    out = []
    for (m, sigma, sp), v in sorted(rows.items(), reverse=True):
        o = np.array([a for a, _ in v]); n = np.array([b for _, b in v])
        out.append(dict(m=m, sigma=sigma, space=sp,
                        obs=o.mean(), obs_se=o.std(ddof=1) / np.sqrt(len(o)),
                        null=n.mean(), null_sd=n.std(ddof=1)))
    json.dump(out, open(OUT, 'w'), indent=1, default=float)

    ms = sorted({r['m'] for r in out})
    for sp in ('subspace', 'raw'):
        print(f'\n=== {sp}: null sd vs m (does it fall like 1/m?) ===')
        print(f'{"m":>6} {"median null sd":>16} {"vs m=256":>10} {"1/m pred":>10}')
        base = None
        for m in ms:
            sds = [r['null_sd'] for r in out if r['m'] == m and r['space'] == sp]
            med = float(np.median(sds))
            base = med if base is None else base
            print(f'{m:6d} {med:16.3e} {base/med:10.2f}x {m/ms[0]:9.1f}x')

        print(f'\n=== {sp}: band-averaged z by t_split and m ===')
        print(f'{"t_split":>8} ' + ' '.join(f'{"m="+str(m):>10}' for m in ms))
        sigs = sorted({r['sigma'] for r in out}, reverse=True)
        for t in (0.925, 0.875, 0.80, 0.70, 0.60, 0.50, 0.40):
            sel = [s for s in sigs if s >= t]
            if not sel:
                continue
            cells = []
            for m in ms:
                z = [r['obs'] / r['null_sd'] for r in out
                     if r['m'] == m and r['space'] == sp
                     and r['sigma'] in sel and r['null_sd'] > 0]
                cells.append(f'{np.mean(z):10.2f}' if z else f'{"-":>10}')
            print(f'{t:8.3f} ' + ' '.join(cells))

dist.barrier()
dist.destroy_process_group()
