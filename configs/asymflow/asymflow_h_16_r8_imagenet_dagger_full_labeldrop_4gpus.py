"""DAGGER variant: label-time CFG dropout (complement_mode='full').

Same as asymflow_h_16_r8_imagenet_dagger_full_4gpus.py EXCEPT the CFG dropout is
applied per captured point at LABEL time rather than per trajectory before
generation (DaggerRolloutHook.label_time_dropout=True):
  - rollouts are ALWAYS class-conditional (no generation-time dropout);
  - each captured point is independently relabelled null w.p. (1 - prob_class) and
    labelled with the whole-pool (unconditional) posterior mean.
Same expected conditional/null proportions and targets as the default; differs only
in the state distribution the null field sees (conditional class-marginal vs
unconditional), which coincide at high sigma (t_split=0.88). Distinct name ->
separate checkpoints / work_dir / wandb run.
"""

_base_ = ['./asymflow_h_16_r8_imagenet_dagger_full_4gpus.py']

name = 'asymflow_h_16_r8_imagenet_dagger_full_labeldrop_4gpus'
work_dir = f'work_dirs/{name}'

# values needed to rebuild custom_hooks (mmcv replaces lists, so the whole list is
# redefined here just to flip label_time_dropout on the DaggerRolloutHook).
latent_size = (3, 256, 256)
round_interval = 500
n_rollout = 1024
rollout_nfe = 50
eval_interval = 500
guidance_scale = 2.3
guidance_interval = [0, 0.88]

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
        type='DaggerRolloutHook',
        round_interval=round_interval,
        n_rollout=n_rollout,
        nfe=rollout_nfe,
        latent_size=latent_size,
        num_classes=1000,
        null_label=1000,
        # prob_class omitted -> inherits train_cfg.prob_class (0.9)
        # t_split omitted -> inherits diffusion.t_split (0.88)
        rollout_chunk=64,
        guidance_scale=1.0,        # unguided (guarded)
        label_time_dropout=True,   # <-- the variant: dropout at label time, per point
        class_sampling='proportional',  # rollout classes ~ dataset prior (like on-path)
        use_ema_rollout=False,
        aggregate_buffer=False,
        priority='NORMAL'),
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
