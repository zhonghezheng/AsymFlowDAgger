"""regft + raw-space MMD with the SOURCE side rolled out from the CFG-guided sampler
(w=2.3), laid out for ONE 8xH100 (80 GB) node.

The guided counterpart of the unguided control raw100_cls8_shboth_4gpus
(asymflow_h_16_r8_imagenet_regft_mmd_sweep_4gpus.py with LAKON_MMD_FEATURE=raw,
LAKON_MMD_CLASSES=8, LAKON_MMD_SHARE=both), differing only in

  * mmd_guidance_scale=2.3 at EVERY band eval (sigma >= 0.875): one batched
    [null; cond] forward, u = u_c + 1.3 (u_c - u_u), gradient through both
    branches. The target side is unchanged (noised real data), so the term trains
    the GUIDED marginal toward the data marginal. NB eval leaves sigma > 0.88
    unguided, so the marginal trained here is not the one inference visits there.
  * the 8xH100 layout below.

Flow matching is unchanged: pure on-path over sigma in (0, 1] (expert=None,
t_split=None, no rollout hook), uniform timesteps carrying the pretraining
logit-normal(0.8, 0.8) mass as a per-point loss weight. Rollout states enter the
MMD only.

Layout. The 4xH200 layout (256/GPU, FM micro-batch 128, mmd_batch 64) peaked at
79.8 GB unguided, and guidance doubles every band eval's batch, so it cannot fit an
80 GB card. Here 128/GPU x 8 keeps the global batch at 1024; the FM runs in exact
micro-batches of 64 (mmd_accum_steps = 128/64 = 2); mmd_batch 32 x 8 ranks keeps
the pooled MMD at 256 trajectories, and mmd_target_n stays 2048 pooled. A guided eval
at mmd_batch 32 costs what an unguided one did at 64. Estimated peak ~51 GB --
extrapolated from the H200 log, not measured.

mmd_classes_per_batch is PER RANK: 4 classes x 8 ranks, 32 trajectories each, gives
8 trajectories per class over 32 pooled classes -- the structure of cls8 on 4 GPUs.

Every knob the sweep config reads from LAKON_* env vars is pinned here, so stray env
vars cannot change the arm. Only the weight stays overridable, and it is in the name.
"""

import os

_base_ = ['./asymflow_h_16_r8_imagenet_regft_mmd_sweep_4gpus.py']

_w = os.environ.get('LAKON_MMD_WEIGHT', '100')
name = f'asymflow_h_16_r8_imagenet_regft_mmd_raw{_w}_cls4_shboth_g2.3_8xh100'
work_dir = f'work_dirs/{name}'

_spg = 128   # per-GPU batch; x 8 GPUs = 1024 global
_mbs = 64    # FM micro-batch (train_cfg.grad_accum_batch_size)

model = dict(diffusion=dict(
    mmd_weight=float(_w),
    mmd_feature='raw',
    mmd_guidance_scale=2.3,
    mmd_guidance_interval=None,   # guided at every rollout eval
    mmd_batch=32,
    # the MMD runs on one micro-batch per iteration; this cancels train_grad_accum's
    # 1/N, so it must equal _spg / _mbs
    mmd_accum_steps=_spg // _mbs,
    mmd_target_n=2048,
    mmd_classes_per_batch=4,
    mmd_target_share='both',
    # 8 ranks x 16 loader threads = the 4 x 32 the H200 runs used per node
    mmd_target_workers=16,
))

train_cfg = dict(grad_accum_batch_size=_mbs)
data = dict(train_dataloader=dict(samples_per_gpu=_spg))

resume_from = f'checkpoints/{name}/latest.pth'
