"""reg_ft control matched on POINTS PER STEP to the MMD arms.

The MMD arms supervise 640 points per GPU per iteration:

    on-path FM   : 256  (samples_per_gpu, full sigma range, true velocity)
    on-policy    : 384  (mmd_batch=64 trajectories x 6 band states at sigma >= 0.875)
    ------------------
    total        : 640

so a plain reg_ft at 256 is not a like-for-like control: any MMD gain could simply be
more gradient signal per step. This arm is reg_ft with samples_per_gpu=640 -- the same
per-step point budget, all of it on-path with the true velocity and no band term at
all. If an MMD arm beats reg_ft-256 but not this, the term bought nothing.

grad_accum_batch_size=128 -> 5 micro-batches, so the per-micro-batch activation
footprint is identical to the MMD arms' (which also split 128) and well inside the
140 GB card; the gradient is exact (accumulate then average).

CAVEAT, stated because it limits what the control proves: matching the point COUNT
also raises the effective batch 2.5x (256 -> 640), which lowers gradient noise
independently of where the points come from. So this brackets the MMD arms rather
than isolating them -- reg_ft-256 and reg_ft-640 together bound what "more points"
alone can deliver.
"""

_base_ = ['./asymflow_h_16_r8_imagenet_regft_4gpus.py']

name = 'asymflow_h_16_r8_imagenet_regft_match640_4gpus'
work_dir = f'work_dirs/{name}'

data = dict(train_dataloader=dict(samples_per_gpu=640))
train_cfg = dict(grad_accum_batch_size=128)   # 5 micro-batches of 128

custom_hooks = [
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('diffusion_ema', ),
        interp_mode='lerp',
        interval=1,
        start_iter=0,
        momentum_policy='fixed',
        interp_cfg=dict(momentum=0.9999),
        priority='VERY_HIGH'),
]
log_config = dict(interval=100, hooks=[
    dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])

resume_from = f'checkpoints/{name}/latest.pth'
