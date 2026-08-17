"""Regular-finetuning (reg_ft) baseline for the DAGGER comparison.

Identical to asymflow_h_16_r8_imagenet_dagger_4gpus.py EXCEPT the DAGGER stream is
fully disabled -- so the only variable between the two runs is the method:
  - expert = None            (no empirical-expert reservoir; frees ~50 GB/rank RAM)
  - diffusion.t_split = None  (on-path FM over the full sigma range)
  - diffusion.onpath_expert_vel = False, roll_weight = 0  (plain FM target)
  - no DaggerRolloutHook      (no rollout / buffer)
Everything else (model arch, released-checkpoint init, optimizer, schedule, data,
eval=10k FID, EMA, wandb + trajectory viz) is inherited unchanged.
"""

_base_ = ['./asymflow_h_16_r8_imagenet_dagger_4gpus.py']

name = 'asymflow_h_16_r8_imagenet_regft_4gpus'
work_dir = f'work_dirs/{name}'

# values needed to (re)build custom_hooks below (mmcv does not expose base vars)
latent_size = (3, 256, 256)
eval_interval = 500
guidance_scale = 2.3
guidance_interval = [0, 0.88]

# disable the DAGGER stream -> plain flow-matching finetuning
model = dict(
    expert=None,
    diffusion=dict(
        t_split=None,
        onpath_expert_vel=False,
        roll_weight=0.0,
    ),
)

# drop the DaggerRolloutHook; keep EMA + trajectory viz for logging parity
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
    dict(
        type='WandbTrajectoryHook',
        interval=eval_interval,
        latent_size=latent_size,
        nfe=16,
        n_samples=4,
        max_cols=8,
        num_classes=1000,
        null_label=1000,
        guidance_scale=guidance_scale,
        guidance_interval=guidance_interval,
        use_ema=True,
        sampler='FlowHeunODE',
        priority='LOW'),
]

# separate wandb run (same project) for side-by-side comparison
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
