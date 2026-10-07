"""Smoke for asymflow_h_16_r8_imagenet_regft_mmd_m2048_4gpus.py at FULL size (2048 pooled
rollouts, 10240 targets, 4 ranks): the MMD from iteration 0, every iteration logged, no
eval, no checkpoints. Measures what the long run cannot afford to discover: peak memory
of the single-pass band graph (512 trajectories) + FM step, s/iter, and that the
in-forward backward coexists with DDP (a sync violation would raise on the first
iteration).
Run on 4 ranks: the sharded kernel and the all-reduces are no-ops at world_size 1.
"""

_base_ = ['./asymflow_h_16_r8_imagenet_regft_mmd_m2048_4gpus.py']

name = 'asymflow_h_16_r8_imagenet_regft_mmd_m2048_smoke'
work_dir = f'work_dirs/{name}'

total_iters = 6

model = dict(diffusion=dict(mmd_start_iter=0, mmd_interval=1))

train_cfg = dict(grad_accum_batch_size=128, log_interval=1)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])

checkpoint_config = dict(interval=100000, by_epoch=False, out_dir='checkpoints/')
evaluation = []
workflow = [('train', total_iters)]
resume_from = None
