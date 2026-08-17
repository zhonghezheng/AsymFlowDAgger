"""Guidance-scale sweep (10k-image FID) for a checkpoint.

Eval-only, run via tools/test.py --ckpt <ckpt>. One GenerativeEvalHook per
guidance_scale in [1.0 .. 3.0] (step 0.25), each generating 10k images with the
Heun step-50 sampler and the [0,0.88] guidance interval (matching how these models
were evaluated during training). Expert disabled (eval uses only the EMA denoiser).
"""

_base_ = ['./asymflow_h_16_r8_imagenet_dagger_4gpus.py']

name = 'asymflow_h_16_r8_imagenet_gsweep'
work_dir = f'work_dirs/{name}'

model = dict(expert=None)                      # eval uses only the EMA denoising weights
data = dict(val=dict(num_test_images=10000))   # 10k generated per guidance scale

step = 50
guidance_interval = [0, 0.88]
guidance_scales = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]

evaluation = [
    dict(
        type='GenerativeEvalHook',
        data='val',
        prefix=f'heun_g{g}_step{step}',
        interval=1,
        sample_kwargs=dict(
            test_cfg_override=dict(
                sampler='FlowHeunODE',
                guidance_scale=g,
                guidance_interval=guidance_interval,
                num_timesteps=step,
            ),
        ),
        feed_batch_size=32,
        metrics=[
            dict(
                type='InceptionMetrics',
                num_images=10000,
                reference_pkl='models/imagenet256_inception_adm.pkl',
                inception_args=dict(
                    type='StyleGAN',
                    inception_path='models/inception-2015-12-05.pt'),
            ),
        ],
        save_best_ckpt=False,
    )
    for g in guidance_scales
]
