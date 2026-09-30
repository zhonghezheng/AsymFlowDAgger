"""reg_ft baseline at lr=1e-5 -- the LR-matched control for the regft_mmd arm.

Identical to asymflow_h_16_r8_imagenet_regft_4gpus.py (plain full-range on-path
flow matching, no expert, no rollout) except lr 2.5e-4 -> 1e-5, matching
asymflow_h_16_r8_imagenet_regft_mmd_4gpus.py. Without this the MMD arm's
training-FID curve has no like-for-like reference: at 1e-5 the weights barely
move, so a flat curve would say nothing about the MMD term. This run isolates
"how much does the model drift at lr=1e-5 with plain FM", and the MMD arm's
deviation from it is the MMD term's actual effect.

wandb-free logging (shared fileset is full) and EMA-only hooks, matching the MMD
arm so the two differ ONLY in the MMD loss.
"""

_base_ = ['./asymflow_h_16_r8_imagenet_regft_4gpus.py']

name = 'asymflow_h_16_r8_imagenet_regft_lr1e5_4gpus'
work_dir = f'work_dirs/{name}'

optimizer = {'diffusion': dict(lr=1e-5)}

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
