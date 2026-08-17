"""Canonical 50k-image FID eval for the DAGGER-comparison checkpoints.

Eval-only, run via tools/test.py --ckpt <ckpt>. The architecture is identical
across the full / project / reg_ft runs, so one config serves all of them -- only
the loaded weights differ. The empirical expert is disabled (val_step runs
diffusion_ema.forward_test, so eval never touches the expert), which also avoids
allocating the per-class reservoir.

Uses the SAME sampler/guidance as the training-time 10k eval
(FlowHeunODE, step-50, g=2.3, guidance_interval=[0,0.88]) so the only difference
from the plotted curves is the sample count (50k vs 10k).
"""

_base_ = ['./asymflow_h_16_r8_imagenet_dagger_4gpus.py']

name = 'asymflow_h_16_r8_imagenet_eval50k'
work_dir = f'work_dirs/{name}'

# eval uses only the EMA denoising weights -> no expert / reservoir needed
model = dict(expert=None)

# generate 50k images (== val dataset length) for the canonical FID
data = dict(val=dict(num_test_images=50000))

guidance_scale = 2.3
guidance_interval = [0, 0.88]
step = 50
prefix = f'heun_g{guidance_scale}({guidance_interval[0]}-{guidance_interval[1]})_step{step}'

evaluation = [
    dict(
        type='GenerativeEvalHook',
        data='val',
        prefix=prefix,
        interval=1,
        sample_kwargs=dict(
            test_cfg_override=dict(
                sampler='FlowHeunODE',
                guidance_scale=guidance_scale,
                guidance_interval=guidance_interval,
                num_timesteps=step,
            ),
        ),
        feed_batch_size=32,
        metrics=[
            dict(
                type='InceptionMetrics',
                num_images=50000,   # canonical FID sample count
                reference_pkl='models/imagenet256_inception_adm.pkl',
                inception_args=dict(
                    type='StyleGAN',
                    inception_path='models/inception-2015-12-05.pt'),
            ),
        ],
        save_best_ckpt=False,
    )
]
