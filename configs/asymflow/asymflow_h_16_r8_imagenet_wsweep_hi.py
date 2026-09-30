"""wsweep EXTENSION: only the high-guidance points w in {2.6, 2.8}.

Backfills the already-evaluated arms (whose 0..2.4 curves are done) at higher CFG,
since FID was still decreasing at 2.4 for the strong arms. Same sampler / interval /
noise stream as asymflow_h_16_r8_imagenet_wsweep.py -- only guidance_settings differ,
so results drop straight into the existing wsweep plot. Eval-only; expert disabled.
"""

import os

_base_ = ['./asymflow_h_16_r8_imagenet_dagger_4gpus.py']

name = 'asymflow_h_16_r8_imagenet_wsweep_hi'
work_dir = f'work_dirs/{name}'

model = dict(expert=None)

num_images = int(os.environ.get('LAKON_EVAL_NUM_IMAGES', 10000))
data = dict(val=dict(num_test_images=num_images))

step = 50
guidance_interval = [0, 0.88]

guidance_settings = [
    (2.6, 'cfg'),
    (2.8, 'cfg'),
]


def _prefix(g, tag):
    if g in (0.0, 1.0):
        return f'{tag}_heun_g{g}_step{step}'
    return f'{tag}_heun_g{g}({guidance_interval[0]}-{guidance_interval[1]})_step{step}'


evaluation = [
    dict(
        type='GenerativeEvalHook',
        data='val',
        prefix=_prefix(g, tag),
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
                num_images=num_images,
                reference_pkl='models/imagenet256_inception_adm.pkl',
                inception_args=dict(
                    type='StyleGAN',
                    inception_path='models/inception-2015-12-05.pt'),
            ),
        ],
        save_best_ckpt=False,
    )
    for g, tag in guidance_settings
]
