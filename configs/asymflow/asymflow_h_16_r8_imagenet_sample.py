"""Sample example images from a checkpoint (no metrics) -- eval-only, tools/test.py.

Generates LAKON_SAMPLE_N images at the guidance scale the arm actually peaks at and
writes them as PNGs via GenerativeEvalHook's viz_dir. Same sampler as every sweep
(FlowHeunODE, step-50, guidance_interval=[0, 0.88]), so the samples correspond to
the settings the FID numbers were measured under.

    LAKON_SAMPLE_W=2.4 python tools/test.py <this> --ckpt <ckpt.pth>
"""

import os

_base_ = ['./asymflow_h_16_r8_imagenet_wsweep.py']

name = 'asymflow_h_16_r8_imagenet_sample'
work_dir = f'work_dirs/{name}'

n = int(os.environ.get('LAKON_SAMPLE_N', 64))
g = float(os.environ.get('LAKON_SAMPLE_W', 2.4))
tag = os.environ.get('LAKON_SAMPLE_TAG', 'sample')
step = 50
guidance_interval = [0, 0.88]

# the inherited val batch is 64; with only `n` images the distributed sampler
# cannot pad a dataset smaller than one batch, so shrink the batch to match.
bs = max(1, min(10, n))
data = dict(val=dict(num_test_images=n),
            val_dataloader=dict(samples_per_gpu=bs),
            test_dataloader=dict(samples_per_gpu=bs))

evaluation = [
    dict(
        type='GenerativeEvalHook',
        data='val',
        prefix=f'{tag}_g{g}',
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
        viz_dir=f'samples/{tag}_g{g}',   # PNGs land here
        viz_num=n,
        metrics=[],                      # samples only; FID at this n is meaningless
        save_best_ckpt=False,
    )
]
