"""Inference with the EMPIRICAL conditional velocity on the noise end (eval only,
tools/test.py [--ckpt]).

Sampler = the run default, Heun step-50, CFG w=2.3 on guidance_interval=[0, 0.88],
except that for sigma > 0.88 -- where CFG is off, so the sampler integrates v_cond --
the model's velocity is REPLACED by the empirical expert's

    v*_cond = (x_t - x0_hat) / sigma,   x0_hat = posterior mean over the row's ENTIRE
                                        class (x2 flips), feat kernel

i.e. the target the band terms regress onto. The Heun step landing on 0.88 takes v* at
its corrector too, so 1 -> 0.88 is integrated with v* alone and the model (with CFG)
takes over from the predictor at 0.88. See lakonlab/models/diffusions/empirical_velocity.py.

One GenerativeEvalHook per arm on the same noise/label stream; 'off' is the unmodified
sampler (the control). Reference FID stats are ImageNet TRAIN (ADM), the same images the
expert's banks hold, so a T=1 expert that hands each trajectory to one training image can
score well by copying -- read FID together with ev_ess / ev_wmax and precision/recall.

Env knobs (defaults in brackets):
  LAKON_EV_ARMS          comma list: 'off' | 'T1' | <temp_spread>   ['off,T1,8']
  LAKON_EV_SWITCH        sigma above which v* drives             ['0.88']
  LAKON_EV_KERNEL        'feat' | 'latent'                       ['feat']
  LAKON_EVAL_NUM_IMAGES  FID sample count (wsweep base)          ['10000']
"""

import os

_base_ = ['./asymflow_h_16_r8_imagenet_wsweep.py']

name = 'asymflow_h_16_r8_imagenet_empvel'
work_dir = f'work_dirs/{name}'

num_images = int(os.environ.get('LAKON_EVAL_NUM_IMAGES', 10000))

step = 50
guidance_scale = 2.3
guidance_interval = [0, 0.88]

_arms = os.environ.get('LAKON_EV_ARMS', 'off,T1,8').split(',')
_switch = float(os.environ.get('LAKON_EV_SWITCH', '0.88'))
_kern = os.environ.get('LAKON_EV_KERNEL', 'feat')
_tag = ('' if _switch == 0.88 else f'_sw{_switch:g}') + ('' if _kern == 'feat' else '_klat')


def _ev(arm):
    if arm == 'off':
        return None
    return dict(sigma_switch=_switch, kernel_space=_kern,
                temp_spread=None if arm == 'T1' else float(arm))


def _prefix(arm):
    base = f'cfg_heun_g{guidance_scale}({guidance_interval[0]}-{guidance_interval[1]})_step{step}'
    return base if arm == 'off' else f'{base}_ev{"T1" if arm == "T1" else "ts" + arm}{_tag}'


evaluation = [
    dict(
        type='GenerativeEvalHook',
        data='val',
        prefix=_prefix(a),
        interval=1,
        sample_kwargs=dict(
            test_cfg_override=dict(
                sampler='FlowHeunODE',
                guidance_scale=guidance_scale,
                guidance_interval=guidance_interval,
                num_timesteps=step,
                empirical_velocity=_ev(a),
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
    for a in _arms
]

# no --ckpt -> the released base weights; never pick up a stray resume checkpoint
resume_from = None
load_from = None
