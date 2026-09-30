"""Unconditional / conditional / CFG eval for a DAGGER-comparison checkpoint.

Eval-only, run via tools/test.py --ckpt <ckpt>. Three GenerativeEvalHooks, all
sharing the Heun step-50 sampler and the same noise/label stream, differing only
in the guidance scale w (CFG: u = u_neg + w * (u_pos - u_neg)):

  w = 0.0  -- UNCONDITIONAL: single forward on the null label (1000). No CFG.
  w = 1.0  -- CONDITIONAL:   single forward on the true class label. No CFG.
  w = 2.3  -- CFG:           batched [null, class] forward with the run's default
              guidance_interval=[0, 0.88] (the setting used for the training-time
              10k FID curves and for eval_fid50k).

The w == 0 branch is handled in LatentDiffusionClassImageMixin.val_step, which
swaps in `negative_labels` -- forward_test itself only ever concatenates the two
branches when w > 1, so w = 0 and w = 1 both cost one network pass per step
(~half the compute of the CFG run).

The architecture is identical across the full / project / reg_ft runs, so one
config serves all of them -- only the loaded weights differ. The empirical expert
is disabled (val_step runs diffusion_ema.forward_test, so eval never touches the
expert), which also avoids allocating the per-class reservoir.

Sample count defaults to 10k (matching the training-time curves and the gsweep);
override from the shell with LAKON_EVAL_NUM_IMAGES=50000 for the canonical FID.
"""

import os

_base_ = ['./asymflow_h_16_r8_imagenet_dagger_4gpus.py']

name = 'asymflow_h_16_r8_imagenet_wsweep'
work_dir = f'work_dirs/{name}'

model = dict(expert=None)  # eval uses only the EMA denoising weights

num_images = int(os.environ.get('LAKON_EVAL_NUM_IMAGES', 10000))
data = dict(val=dict(num_test_images=num_images))

step = 50
guidance_interval = [0, 0.88]
default_guidance_scale = 2.3  # the run default (see eval_fid50k / dagger config)

# Swept guidance scales, env-overridable so the sweep can be re-pointed without
# editing this (shared) config: LAKON_GUIDANCE='0,1,1.6,1.8,2.0,2.2,2.4,2.6,2.8'
# restores the previous 9-point set. Default is 0, 1, then 1.8..2.6 -- 1.6 and 2.8
# are dropped because every recorded optimum sits at 2.0-2.6, so those two ends only
# cost wall time (7 points ~ 58 min vs ~1h26 for 9, since each CFG point is 2 net
# passes per step and w in {0, 1} is 1).
# The tag is derived, not listed: w=0 is unconditional (null labels), w=1 is
# conditional with no CFG branch, everything else is CFG. guidance_interval is
# irrelevant for the first two (no CFG branch is taken), so it is only tagged into
# the CFG prefix.
_gvals = os.environ.get('LAKON_GUIDANCE', '0,1,1.8,2.0,2.2,2.4,2.6')
guidance_settings = [
    (float(g), 'uncond' if float(g) == 0.0 else 'cond' if float(g) == 1.0 else 'cfg')
    for g in _gvals.split(',')
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
