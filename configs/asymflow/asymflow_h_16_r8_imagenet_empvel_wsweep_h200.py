"""EMPIRICAL conditional velocity on a high-sigma window + a CFG-w sweep below 0.88, for
della's H200 nodes (eval only, tools/test.py [--ckpt]; launched by
sbatch_scripts/eval_empvel_wsweep.sh, one job per window by
sbatch_scripts/submit_empvel_sweep.sh).

Each arm is one Heun step-50 trajectory (sigma grid 1.00, 0.98, .., 0.02). On its v*
window [lo, hi] the model's velocity is replaced by

    v*_cond = (x_t - x0_hat) / sigma,   x0_hat = posterior mean over the row's ENTIRE
                                        class (x2 flips) -- the band terms' target

elsewhere the model runs: conditional only above 0.88, CFG weight w on
guidance_interval=[0, 0.88]. Windows (lo-hi), default all four:

  off        the unmodified sampler -- the control, no banks built
  0.94-1     v* 1 -> 0.94, model (cond) 0.94 -> 0.88, CFG w below
  0.88-1     v* 1 -> 0.88,                            CFG w below
  0.88-0.94  model (cond) 1 -> 0.94, v* 0.94 -> 0.88, CFG w below

The window is cut by Heun STEP, not by eval (EmpiricalVelocity sigma_switch / sigma_hi):
the steps starting at hi .. lo+0.02 are v*'s, both evals, so v* integrates exactly
hi -> lo. See lakonlab/models/diffusions/empirical_velocity.py, and
asymflow_h_16_r8_imagenet_empvel.py for the Brev/H100 version (w=2.3, window 0.88-1).

w follows the wsweep convention, restricted to what the override can run with:
  w = 1  -- no CFG anywhere (one forward per step)
  w > 1  -- CFG on [0, 0.88]
w = 0 is refused: val_step then samples on the null label for the WHOLE trajectory,
and v*_cond needs the real class. 0 < w < 1 is refused too: forward_test only takes
the CFG branch for w > 1, so it would silently run as w = 1.

The control's prefix is exactly the wsweep's, and the val noise is seeded per image
index (ImageNet dataset, idx + 1000), so it also lines up with any wsweep of the same
checkpoint and image count. Each v* arm's prefix carries '_ev<T>(<lo>-<hi>)'.

Reference FID stats are ImageNet TRAIN (ADM), the images the banks hold. At T=1 the
posterior is a near-delta by sigma 0.94 (smoke: ev_ess 1.03, ev_wmax 0.989), i.e.
the trajectory is handed to ONE training image -- read FID together with ev_ess /
ev_wmax and precision/recall, and consider a temp_spread arm.

Env knobs (defaults in brackets):
  LAKON_GUIDANCE         comma list of w, each >= 1       ['1,1.8,2.0,2.2,2.4,2.6']
  LAKON_EV_WINDOWS       comma list of 'off' | '<lo>-<hi>'
                                              ['off,0.94-1,0.88-1,0.88-0.94']
  LAKON_EV_ARMS          comma list: 'T1' | <temp_spread> ['T1']
  LAKON_EV_KERNEL        'feat' | 'latent'                ['feat']
  LAKON_U8_CACHE         u8 image cache prefix; '' forces JPEGs
                         [/dev/shm/asymflow/train_u8_256 if complete, else
                          data/cached_latents/train_u8_256 if complete]
  LAKON_EV_READ_THREADS  pread threads per rank for the banks  ['32']
  LAKON_EV_TAU2          Gaussian-blob (KDE) prior variance per kernel dim, tag _kde<tau2>  [off]
  LAKON_CFG_INTERVAL     '<lo>-<hi>' sigma range where CFG w applies  ['0-0.88']
                         (a v* window overrides the guided velocity inside it)
  LAKON_EVAL_NUM_IMAGES  FID sample count (wsweep base)   ['10000']
"""

import os

_base_ = ['./asymflow_h_16_r8_imagenet_wsweep.py']

name = 'asymflow_h_16_r8_imagenet_empvel_wsweep_h200'
work_dir = f'work_dirs/{name}'

num_images = int(os.environ.get('LAKON_EVAL_NUM_IMAGES', 10000))  # as the wsweep base

step = 50
guidance_interval = [float(v) for v in os.environ.get('LAKON_CFG_INTERVAL', '0-0.88').split('-')]
assert len(guidance_interval) == 2 and 0.0 <= guidance_interval[0] < guidance_interval[1] <= 1.0, \
    guidance_interval

_gvals = [float(g) for g in os.environ.get('LAKON_GUIDANCE', '1,1.8,2.0,2.2,2.4,2.6').split(',')]
assert all(g >= 1.0 for g in _gvals), (
    f'LAKON_GUIDANCE={_gvals}: every w must be >= 1. w = 0 samples on the null label '
    'throughout (v*_cond needs the class), and 0 < w < 1 runs as w = 1.')
# v* windows as (lo, hi), or None for the control
_windows = [None if w == 'off' else tuple(float(v) for v in w.split('-'))
            for w in os.environ.get('LAKON_EV_WINDOWS', 'off,0.94-1,0.88-1,0.88-0.94').split(',')]
assert all(w is None or (len(w) == 2 and 0.0 <= w[0] < w[1] <= 1.0) for w in _windows), _windows
_arms = os.environ.get('LAKON_EV_ARMS', 'T1').split(',')
assert 'off' not in _arms, "the control is the 'off' WINDOW (LAKON_EV_WINDOWS), not an arm"
_kern = os.environ.get('LAKON_EV_KERNEL', 'feat')
_tau2 = os.environ.get('LAKON_EV_TAU2')
_tau2 = float(_tau2) if _tau2 else None

# della has no node-local disk on ailab; its verified copy of the cache sits on GPFS.
# The banks are every row's ENTIRE class, so without a cache each eval batch decodes
# ~80k JPEGs per rank. The log line 'u8 image cache in use' confirms which one is read.
_u8 = os.environ.get('LAKON_U8_CACHE')
if _u8 is None:
    _u8 = next((p for p in ('/dev/shm/asymflow/train_u8_256', 'data/cached_latents/train_u8_256')
                if os.path.exists(p + '.complete')), '')
_u8 = _u8 or None
_read_threads = int(os.environ.get('LAKON_EV_READ_THREADS', 32))


def _ev(arm, win):
    if win is None:
        return None
    return dict(sigma_switch=win[0], sigma_hi=win[1], kernel_space=_kern,
                temp_spread=None if arm == 'T1' else float(arm), kde_tau2=_tau2,
                u8_cache=_u8, u8_read_threads=_read_threads)


def _prefix(g, arm, win):
    # the control's prefix is exactly the wsweep's
    base = (f'cond_heun_g{g}_step{step}' if g == 1.0 else
            f'cfg_heun_g{g}({guidance_interval[0]:g}-{guidance_interval[1]:g})_step{step}')
    if win is None:
        return base
    return (f'{base}_ev{"T1" if arm == "T1" else "ts" + arm}({win[0]:g}-{win[1]:g})'
            + ('' if _kern == 'feat' else '_klat') + ('' if _tau2 is None else f'_kde{_tau2:g}'))


# one hook per distinct prefix: the control is the same for every temperature arm
_points = list({_prefix(g, a, win): (g, a, win)
                for g in _gvals for win in _windows for a in _arms}.items())

evaluation = [
    dict(
        type='GenerativeEvalHook',
        data='val',
        prefix=p,
        interval=1,
        sample_kwargs=dict(
            test_cfg_override=dict(
                sampler='FlowHeunODE',
                guidance_scale=g,
                guidance_interval=guidance_interval,
                num_timesteps=step,
                empirical_velocity=_ev(a, win),
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
    for p, (g, a, win) in _points
]

# no --ckpt -> the released base weights; never pick up a stray resume checkpoint
resume_from = None
load_from = None
