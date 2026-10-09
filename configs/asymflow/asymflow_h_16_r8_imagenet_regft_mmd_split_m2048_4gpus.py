"""regft + SPLIT MMD on the band sigma >= 0.88: one term on the rank-8 subspace features,
one on the complement x - project_fn(x), at 2048 pooled rollouts against 10240 pooled
targets, on 4 GPUs.

The training analogue of the split MMD guidance (MMDGuidance feature='split', the first
guidance arm to reach the unguided baseline: 4.325 vs 4.331 at alpha 3e-4, 10k FID).
Two arms, LAKON_MMD_SPLIT:

  sum   the plain sum MMD^2_sub + MMD^2_comp and its exact gradient
        (mmd_split_norm=False). Weight 100, as the raw / subspace arms.
  norm  each part's state gradient rescaled to that part's own pooled norm, the split
        guidance step, backpropagated as a unit-step regression of the rollout onto
        its own guided state (mmd_split_norm=True; see GaussianFlowMMD). Its weight is
        on a different scale (it acts as the guidance alpha): set it so the term's
        parameter gradient (logged mmd_pgrad_norm) matches the sum arm's.

Everything else is asymflow_h_16_r8_imagenet_regft_mmd_m2048_4gpus.py (band sigma 0.98 ..
0.88, 6 Heun steps, unguided rollout, 512 rollouts/rank x 4, 64 classes/rank x 8
trajectories, 5 class-matched targets per rollout, full-fp32 sharded kernel, 5000 iters,
MMD from iter 500, the chunked in-forward backward), except: the kernel and the target
sharing, taken from the guidance arm -- ONE Gaussian width, 3x the mean target pairwise
d2 (the power test's plateau), and mmd_target_share='bank' (one image draw per band,
noise redrawn per step) -- and the LR, 1e-4 instead of the base's 2.5e-4 (pretraining
used 2e-4, constant).

LAKON_LR sets the peak learning rate (default 1e-4). It is ALWAYS in the name ('_lr<x>'),
so these runs never resume from the earlier untagged 2.5e-4 checkpoints.
LAKON_MMD_CHUNK=128 replays the band in chunks (same gradient; an 80 GB H100 may not hold
the single-pass 512-trajectory graph). LAKON_MMD_WEIGHT overrides the weight; it is in
the name.
"""

import os

_base_ = ['./asymflow_h_16_r8_imagenet_regft_mmd_m2048_4gpus.py']

_mode = os.environ.get('LAKON_MMD_SPLIT', 'sum')
assert _mode in ('sum', 'norm'), _mode
_w = os.environ.get('LAKON_MMD_WEIGHT', '100' if _mode == 'sum' else None)
assert _w is not None, (
    "LAKON_MMD_SPLIT=norm needs LAKON_MMD_WEIGHT: its weight is on the guidance-alpha "
    'scale, calibrated against the sum arm by mmd_pgrad_norm.')
# LAKON_LR: peak learning rate, default 1e-4 (the base config's 2.5e-4 was 1.25x
# pretraining's constant 2e-4). Always in the name.
_lr = float(os.environ.get('LAKON_LR', '1e-4'))
name = (f'asymflow_h_16_r8_imagenet_regft_mmd_split{_mode}{_w}_bw3_cls8_shbank'
        f'_m2048_n10240_lr{_lr:g}_4gpus')
work_dir = f'work_dirs/{name}'

model = dict(diffusion=dict(
    mmd_weight=float(_w),
    mmd_feature='split',
    mmd_split_norm=_mode == 'norm',
    mmd_bandwidths=(3.0, ),
    mmd_target_share='bank',
))

resume_from = f'checkpoints/{name}/latest.pth'

optimizer = {'diffusion': dict(lr=_lr)}
