"""CFG guidance-INTERVAL sweep for a DAGGER-comparison checkpoint.

Eval-only (tools/test.py --ckpt <ckpt>). Fixes the sampler (Heun step-50) and the
guidance scale, and sweeps the sigma window [lo, hi] over which CFG is applied:

    forward_u gates CFG per step by   lo <= sigma <= hi
    (see gaussian_flow.py: guidance_active = (t>=lo) & (t<=hi)).

The run default is [0, 0.88] -- CFG is switched OFF for the noisiest steps
(sigma > 0.88). Extending hi toward 1.0 turns guidance back on at high noise;
lowering hi restricts guidance to the low-noise tail. This is the region a
t_split=0.92 DAGGER model already rewrote, so the two arms (reg_ft vs bank128)
are expected to prefer different cutoffs.

Grid = guidance_scales x interval upper-cutoffs (lower bound fixed at 0). One
GenerativeEvalHook per combo, all sharing the same noise/label stream. As in the
wsweep config the empirical expert is disabled (eval runs the EMA denoiser).

num_images defaults to 10k (matches the wsweep / training FID curves);
override with LAKON_EVAL_NUM_IMAGES=50000 for canonical FID.
"""

import os

_base_ = ['./asymflow_h_16_r8_imagenet_dagger_4gpus.py']

name = 'asymflow_h_16_r8_imagenet_isweep'
work_dir = f'work_dirs/{name}'

model = dict(expert=None)  # eval uses only the EMA denoising weights

num_images = int(os.environ.get('LAKON_EVAL_NUM_IMAGES', 10000))
data = dict(val=dict(num_test_images=num_images))

step = 50

# swept axes: guidance scale x interval upper cutoff (lo fixed at 0)
guidance_scales = [2.0, 2.4]
interval_his = [0.85, 0.92]


def _prefix(g, hi):
    return f'cfg_g{g}_i0-{hi}_heun_step{step}'


evaluation = [
    dict(
        type='GenerativeEvalHook',
        data='val',
        prefix=_prefix(g, hi),
        interval=1,
        sample_kwargs=dict(
            test_cfg_override=dict(
                sampler='FlowHeunODE',
                guidance_scale=g,
                guidance_interval=[0, hi],
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
    for g in guidance_scales
    for hi in interval_his
]
