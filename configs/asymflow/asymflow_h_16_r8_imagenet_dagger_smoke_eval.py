"""Smoke test WITH one eval loop (generation + InceptionMetrics/FID + wandb).

Inherits the DAGGER smoke config and re-enables a single, cheap evaluation to
exercise the eval path end to end: generate a few images, compute FID against the
reference stats, and log the metric to (offline) wandb alongside the training
losses and trajectory grid. FID over 256 images is numerically meaningless -- this
only validates that the eval pipeline runs and logs.
"""

_base_ = ['./asymflow_h_16_r8_imagenet_dagger_smoke.py']

name = 'asymflow_h_16_r8_imagenet_dagger_smoke_eval'
work_dir = f'work_dirs/{name}'

# eval generation count == len(val dataset) == num_test_images. Cap it to 10k so
# the eval generates 10k images (not the 50k default) for the FID.
data = dict(val=dict(num_test_images=10000))

# offline wandb (compute nodes have no internet to api.wandb.ai); logs to
# work_dir-local wandb/, sync to wandb.ai from the login node afterward.
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(project='asymflow-dagger', name=name, mode='offline')),
    ],
)

evaluation = [
    dict(
        type='GenerativeEvalHook',
        data='val',
        prefix='smoke',
        interval=40,   # fire once at iter 40 (total_iters=60), so training resumes after
        sample_kwargs=dict(
            test_cfg_override=dict(
                sampler='FlowHeunODE',
                guidance_scale=2.3,
                guidance_interval=[0, 0.88],
                num_timesteps=8,   # few steps for speed
            ),
        ),
        feed_batch_size=32,
        metrics=[
            dict(
                type='InceptionMetrics',
                num_images=10000,   # use all 10k generated images for the FID
                reference_pkl='models/imagenet256_inception_adm.pkl',
                inception_args=dict(
                    type='StyleGAN',
                    inception_path='models/inception-2015-12-05.pt'),
            ),
        ],
        save_best_ckpt=False,
    )
]
