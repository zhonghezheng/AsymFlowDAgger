"""Smoke for asymflow_h_16_r8_imagenet_regft_mmd_split_m2048_4gpus.py at FULL size (2048
pooled rollouts, 10240 targets, 4 ranks): the MMD from iteration 0, every iteration
logged, no eval, no checkpoints. Measures peak memory, s/iter, mmd_replay_err, and the
term's parameter-gradient norm (mmd_pgrad_norm) -- the number the norm arm's weight is
calibrated by. LAKON_MMD_SPLIT / LAKON_MMD_WEIGHT / LAKON_MMD_CHUNK as the run config.
"""

import os

_base_ = ['./asymflow_h_16_r8_imagenet_regft_mmd_split_m2048_4gpus.py']

# per arm, so the two smokes can run side by side
name = f"asymflow_h_16_r8_imagenet_regft_mmd_split{os.environ.get('LAKON_MMD_SPLIT', 'sum')}_m2048_smoke"
work_dir = f'work_dirs/{name}'

total_iters = 6

model = dict(diffusion=dict(mmd_start_iter=0, mmd_interval=1))

train_cfg = dict(grad_accum_batch_size=128, log_interval=1)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])

checkpoint_config = dict(interval=100000, by_epoch=False, out_dir='checkpoints/')
evaluation = []
workflow = [('train', total_iters)]
resume_from = None
