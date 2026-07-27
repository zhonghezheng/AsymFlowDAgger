"""DAGGER ablation: complement_mode='full'.

Identical to asymflow_h_16_r8_imagenet_dagger_4gpus.py EXCEPT the expert target
keeps the complement noise: L = (x_t - x0_hat)/sigma_clamped (plain velocity toward
the posterior mean) instead of the 'project' target (Pε - x0_hat, complement
dropped). Tests whether the projected/complement-dropped target drives the
divergence. Distinct name -> separate checkpoints / work_dir / wandb run.
"""

_base_ = ['./asymflow_h_16_r8_imagenet_dagger_4gpus.py']

name = 'asymflow_h_16_r8_imagenet_dagger_full_4gpus'
work_dir = f'work_dirs/{name}'

model = dict(diffusion=dict(complement_mode='full'))

# override so the wandb run name matches this variant (log_config is inherited
# with the base name baked in otherwise)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(project='asymflow-dagger', name=name, mode='offline')),
    ])

resume_from = f'checkpoints/{name}/latest.pth'
