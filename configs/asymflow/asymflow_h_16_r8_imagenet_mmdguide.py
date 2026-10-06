"""Inference-time MMD guidance sweep for one checkpoint (eval only, tools/test.py --ckpt).

The sampler is the run default -- Heun step-50, CFG w=2.3 on guidance_interval=[0, 0.88]
-- and at every one of its 50 landing states (sigma = 1.00 .. 0.02, before the
predictor eval) each sample takes one normalized gradient step down the MMD^2 between
ITSELF (a point mass) and K real images of its own class noised to that sigma:

    g_i = d MMD^2(delta_{x_i}, {(1 - sigma) x_0^k + sigma eps^k}_k) / d x_i
    x_i' = x_i - alpha * g_i / ||g_i||_2 * ||x_i||_2
    x_i <- x_i' / ||x_i'||_2 * ||x_i||_2            (renormalized to the pre-step norm)

so a sample's perturbation depends only on the target distribution, not on the other
rollouts. LAKON_MMDG_OBJ=pooled restores the first version (one MMD between the
pooled rollout batch and the pooled targets, norms over the pooled batch).

See lakonlab/models/diffusions/mmd_guidance.py. One GenerativeEvalHook per alpha;
alpha = 0 is the unguided baseline (no MMD code runs), so every sweep carries its own
reference FID on the same noise/label stream.

Env knobs (defaults in brackets):
  LAKON_MMDG_ALPHA       comma list of alphas            ['0,0.001,0.003,0.01']
  LAKON_MMDG_OBJ         'sample' | 'pooled'             ['sample']
  LAKON_MMDG_WINDOW      'lo,hi' sigma window guided     ['0,1'  = all 50 steps]
  LAKON_MMDG_NORM        'sample' | 'batch'              [per objective: sample / batch]
  LAKON_MMDG_RENORM      1 = restore ||x|| after a step  ['1']
  LAKON_MMDG_TGT         real images per sample per step [per objective: 16 / 4]
  LAKON_MMDG_SHARE       'none' | 'bank' | 'both'        ['both']
  LAKON_MMDG_BS          rollouts per GPU per batch      ['64'  -> 512 pooled]
  LAKON_MMDG_BW          kernel widths: 'mean' | 'spread' ['mean']
  LAKON_MMDG_FEAT        MMD space: 'raw' | 'subspace'   ['raw']
  LAKON_MMDG_KERNEL      'rbf' | 'energy' (distance kernel, no bandwidth) ['rbf']
  LAKON_MMDG_BWS         RBF width multipliers, comma list ['0.25,0.5,1,2,4']
  LAKON_MMDG_KNORM       1 = per-width normalised mixture ['0']
  LAKON_MMDG_EBETA       energy-kernel distance exponent, (0, 2) ['1']
  LAKON_MMDG_MEASURE     1 = ONE unguided arm that only MEASURES the MMD^2 at every
                         one of the 50 states (value + square per sigma, for error
                         bars across batches); the alpha list is ignored   ['0']
  LAKON_EVAL_NUM_IMAGES  FID sample count (wsweep base)  ['10000']
Off-default knobs are tagged into the prefix.
"""

import os

_base_ = ['./asymflow_h_16_r8_imagenet_wsweep.py']

name = 'asymflow_h_16_r8_imagenet_mmdguide'
work_dir = f'work_dirs/{name}'

num_images = int(os.environ.get('LAKON_EVAL_NUM_IMAGES', 10000))  # as the wsweep base

step = 50
guidance_scale = 2.3
guidance_interval = [0, 0.88]

_alphas = [float(a) for a in os.environ.get('LAKON_MMDG_ALPHA', '0,0.001,0.003,0.01').split(',')]
_win = tuple(float(v) for v in os.environ.get('LAKON_MMDG_WINDOW', '0,1').split(','))
_obj = os.environ.get('LAKON_MMDG_OBJ', 'sample')
_norm_def, _tgt_def = ('sample', 16) if _obj == 'sample' else ('batch', 4)
_norm = os.environ.get('LAKON_MMDG_NORM', _norm_def)
_renorm = os.environ.get('LAKON_MMDG_RENORM', '1') != '0'
_tgt = int(os.environ.get('LAKON_MMDG_TGT', _tgt_def))
# Default 'both': each sample's target images AND their noise are drawn once per batch
# and reused at every step, so every target is a straight line (1-sigma) x0 + sigma eps.
# 'none' (fresh images and noise every step) was the default of the 2026-10-02 sweeps.
_share = os.environ.get('LAKON_MMDG_SHARE', 'both')
# Rollouts per GPU per batch (the eval batch). The POOLED objective gathers every rank's
# batch, so this x 8 GPUs is the rollout side of the MMD: 64 -> 512 rollouts, whose
# finite-sample noise (and the cross-rollout repulsion in the gradient) the target count
# cannot reduce. Tagged '_bs<n>' off the default.
_bs = int(os.environ.get('LAKON_MMDG_BS', 64))
# tools/test.py builds the eval loader from test_dataloader (val_dataloader is the
# training-time eval's), so both are set
data = dict(val_dataloader=dict(samples_per_gpu=_bs), test_dataloader=dict(samples_per_gpu=_bs))
# kernel widths as multiples of the target pairwise d2's MEAN (original) or its SPREAD
# (standard deviation) -- see mmd2_rbf. Tagged '_bwsp' for spread.
_bw = os.environ.get('LAKON_MMDG_BW', 'mean')
# space the MMD is computed in: raw flattened latents, or the rank-8 subspace features
# (feat_fn, 2048-d) -- the training term's mmd_feature. Tagged '_fsub'.
_feat = os.environ.get('LAKON_MMDG_FEAT', 'raw')
# kernel: the RBF mixture, or the energy-distance kernel -||a - b|| (no bandwidth; the
# width knob does not apply). Tagged '_ken'.
_kern = os.environ.get('LAKON_MMDG_KERNEL', 'rbf')
# RBF width multipliers (x the mean target d2, or x its sd under BW=spread), and the
# per-width normalisation that keeps a wide-ranging mixture balanced. Tagged
# '_bw<a>-<b>-..' when off the default list, '_kn' when normalised.
_bws_def = (0.25, 0.5, 1.0, 2.0, 4.0)
_bws = tuple(float(v) for v in os.environ.get('LAKON_MMDG_BWS', '0.25,0.5,1,2,4').split(','))
_knorm = os.environ.get('LAKON_MMDG_KNORM', '0') != '0'
# energy kernel's distance exponent beta (k = -||a - b||^beta). Tagged '_eb<beta>'.
_ebeta = float(os.environ.get('LAKON_MMDG_EBETA', '1'))
# NB the first sweep (2026-10-02, pooled objective) ran before the objective existed,
# so its untagged '_mmdg<alpha>' prefixes are the POOLED arm; per-sample runs are '_ps'.
# The share tag is still keyed on 'none', so those old prefixes stay distinct ('_shboth').
_tag = (('_ps' if _obj == 'sample' else '')
        + ('' if _win == (0.0, 1.0) else f'_w{_win[0]:g}-{_win[1]:g}')
        + ('' if _norm == _norm_def else f'_n{_norm}')
        + ('' if _renorm else '_nr')
        + ('' if _tgt == _tgt_def else f'_tg{_tgt}')
        + ('' if _share == 'none' else f'_sh{_share}')
        + ('' if _bs == 64 else f'_bs{_bs}')
        + ('' if _bw == 'mean' else '_bwsp')
        + ('' if _feat == 'raw' else '_fsub')
        + ('' if _kern == 'rbf' else '_ken')
        + ('' if _bws == _bws_def else '_bw' + '-'.join(f'{v:g}' for v in _bws))
        + ('_kn' if _knorm else '')
        + ('' if _ebeta == 1.0 else f'_eb{_ebeta:g}'))


_measure = os.environ.get('LAKON_MMDG_MEASURE', '0') != '0'
if _measure:
    _alphas = [0.0]


def _mmd_guide(alpha):
    if alpha == 0 and not _measure:
        return None
    return dict(scale=alpha, measure_only=_measure, objective=_obj, sigma_range=_win, norm=_norm, renorm=_renorm,
                target_per_row=_tgt, target_share=_share, width=_bw, feature=_feat,
                kernel=_kern, bandwidths=_bws, normalize=_knorm, energy_beta=_ebeta)


def _prefix(alpha):
    base = f'cfg_heun_g{guidance_scale}({guidance_interval[0]}-{guidance_interval[1]})_step{step}'
    if _measure:
        return f'{base}_mmdmeas{_tag}'
    return base if alpha == 0 else f'{base}_mmdg{alpha:g}{_tag}'


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
                mmd_guide=_mmd_guide(a),
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
    for a in _alphas
]

# no --ckpt -> the released base weights (denoising.pretrained, cloned into the EMA);
# never pick up a stray resume checkpoint from the inherited run name
resume_from = None
load_from = None
