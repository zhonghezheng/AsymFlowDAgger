"""Smoke-test config for DAGGER AsymFlow finetuning.

Inherits the real 4-GPU DAGGER config and shrinks everything for a fast,
memory-safe end-to-end check: tiny per-class reservoir, no torch.compile, a
short run that still fires a couple of rollout rounds, and light dataloaders.
NOT a real training run -- just verifies the pipeline loads the released
checkpoint and runs on-path + rollout/expert + buffer steps without crashing.

Launch with --no-validate (eval is off for the smoke test anyway).
"""

_base_ = ['./asymflow_h_16_r8_imagenet_dagger_4gpus.py']

name = 'asymflow_h_16_r8_imagenet_dagger_smoke'
work_dir = f'work_dirs/{name}'

# short run that still crosses a rollout round (round_interval below)
total_iters = 60

model = dict(
    expert=dict(
        bank_size=16,       # tiny disk bank per class for a fast smoke
        num_workers=8,
    ),
    diffusion=dict(
        denoising=dict(
            compile_forward=False,  # skip the (minutes-long) compile for a smoke test
        ),
    ),
)

# lighter dataloaders to keep host RAM modest
data = dict(
    workers_per_gpu=4,
    prefetch_factor=2,
)

# frequent logging so the smoke run is visibly progressing. wandb in OFFLINE mode
# for the smoke test (logs to work_dir/wandb, no internet needed); the real config
# uses online. Verifies training-loss + trajectory image logging end to end.
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(project='asymflow-dagger-smoke', name=name, mode='offline')),
    ],
)

# redefine hooks (lists are replaced, not merged): EMA + a small, frequent
# DAGGER round so the rollout/expert/buffer path is exercised within 60 iters.
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
        round_interval=20,     # fire rollout rounds at iter 20, 40, 60
        n_rollout=32,          # small rollout for speed
        nfe=8,                 # few Euler steps
        latent_size=(3, 256, 256),
        num_classes=1000,
        null_label=1000,
        rollout_chunk=32,
        guidance_scale=1.0,              # unguided (guarded)
        label_time_dropout=True,
        class_sampling='proportional',
        use_ema_rollout=False,
        aggregate_buffer=False,
        priority='NORMAL'),
    # log a denoising-trajectory grid to (offline) wandb; small + frequent for the test
    dict(
        type='WandbTrajectoryHook',
        interval=20,
        latent_size=(3, 256, 256),
        nfe=8,
        n_samples=4,
        num_classes=1000,
        null_label=1000,
        guidance_scale=2.3,
        guidance_interval=[0, 0.88],
        use_ema=True,
        sampler='FlowHeunODE',
        priority='LOW'),
]

# don't save real checkpoints during the smoke test
checkpoint_config = dict(interval=100000, by_epoch=False, out_dir='checkpoints/')

# no evaluation for the smoke test. (Also avoids a train.py quirk: with
# --no-validate the eval dataloaders are never attached, but the dataloader-warmup
# loop still iterates cfg.evaluation and reads eval_cfg.dataloader -> AttributeError.
# An empty list makes that loop a no-op.)
evaluation = []

# start fresh from the released checkpoint (never resume a real run)
resume_from = None
