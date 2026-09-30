"""regft + MMD on the ORIGINAL band [0.92, 1.0], with an enlarged CLASS-MATCHED target.

Identical to asymflow_h_16_r8_imagenet_regft_mmd_sweep_4gpus.py except the target
side of the MMD is drawn from mmd_target_n=2048 samples instead of the 256 the
rollout supplies. MMD does not require m == n, and the target side costs nothing:
no network forward, no graph, no gradient.

Exact and nearly free, because feat_fn is LINEAR and proj_buffer is ORTHONORMAL:
    feat((1-s)x_0 + s*eps) = (1-s)feat(x_0) + s*feat(eps),  feat(eps) ~ N(0, I)
so target samples are generated straight from a cache of past feat(x_0) rows (2048-d,
~8 KB each) with the noise drawn in feature space. Both identities verified.

CLASS MATCHING is the part that matters. The rollout covers only ~64 of 1000 classes
per estimate; drawing the target from the data pool at large makes MMD^2 read +1.0e-3
on a PERFECT model -- 10x the noise floor, ~40% of a real signal -- and that bias does
not shrink with n, since it is a class-coverage artifact the model cannot remove.
Training against it would push each class to broaden toward the full-class mixture.
So the cache is keyed BY CLASS and each row's target rows are drawn from its own
class; null-labelled (CFG-dropout) rows are the exception, their rollout being the
unconditional marginal, so they draw from all classes.

Expected gain, measured on class-structured synthetic data with a perfect model:
floor sd 2.1e-4 at n=256 -> 7.6e-5 at n=2048, ~2.8x, with the mean correctly at ~0.
Bounded by the rollout side: the estimator keeps a 1/(m(m-1)) term, so mmd_batch
stays the binding constraint.
"""

import os

_base_ = ['./asymflow_h_16_r8_imagenet_regft_mmd_sweep_4gpus.py']

_w = os.environ.get('LAKON_MMD_WEIGHT', '100')
_n = os.environ.get('LAKON_MMD_TARGET_N', '2048')
_feat = os.environ.get('LAKON_MMD_FEATURE', 'subspace')
_tag = 'sub' if _feat == 'subspace' else _feat

name = f'asymflow_h_16_r8_imagenet_regft_mmd_tgt{_n}_{_tag}{_w}_4gpus'
work_dir = f'work_dirs/{name}'

model = dict(diffusion=dict(
    mmd_weight=float(_w),
    mmd_feature=_feat,
    mmd_t_split=0.92,        # the original band, [0.92, 1.0]
    mmd_t_hi=None,           # no upper edge -> full backprop from sigma=1
    mmd_target_n=int(_n),    # pooled target samples, class-matched
))

resume_from = f'checkpoints/{name}/latest.pth'
