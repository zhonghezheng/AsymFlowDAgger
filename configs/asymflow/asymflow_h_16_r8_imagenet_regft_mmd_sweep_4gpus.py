"""regft_mmd with the MMD weight AND feature space swept, one run per (feature, w).

Same arm as asymflow_h_16_r8_imagenet_regft_mmd_4gpus.py in every respect -- band
sigma >= 0.875, mmd_batch=64 pooled to 256 across ranks, bandwidths
(0.25 .. 4.0) on the median heuristic, lr 1e-5, grad-accum 128 -- with only
mmd_weight and mmd_feature varying.

NB the name carries BOTH, unlike the two original configs whose names are fixed:
a second weight run under a fixed name would resume from the first run's
iter_5000.pth (resume_from points at checkpoints/<name>/latest.pth) and terminate
immediately, having already 'reached' 5000 iters.

The original w=100 runs live at the unsuffixed names (regft_mmd_4gpus,
regft_mmd_raw_4gpus) and are untouched by this config.
"""

import os

_base_ = ['./asymflow_h_16_r8_imagenet_regft_mmd_4gpus.py']

_w = os.environ.get('LAKON_MMD_WEIGHT', '100')
_feat = os.environ.get('LAKON_MMD_FEATURE', 'subspace')
_tag = 'sub' if _feat == 'subspace' else _feat

_cls = os.environ.get('LAKON_MMD_CLASSES')
# How much of the target draw is reused across the band's steps: 'none' (fresh
# images and noise per step -- what every arm so far ran), 'bank' (images held,
# noise per step) or 'both' (images and noise held, so the target set is a set of
# straight-line trajectories mirroring the rollout's structure).
_share = os.environ.get('LAKON_MMD_SHARE', 'none')
name = 'asymflow_h_16_r8_imagenet_regft_mmd_' + _tag + _w \
    + (f'_cls{_cls}' if _cls else '') \
    + ('' if _share == 'none' else f'_sh{_share}') + '_4gpus'
work_dir = f'work_dirs/{name}'

model = dict(diffusion=dict(
    mmd_weight=float(_w),
    mmd_feature=_feat,
    # 64 rollout trajectories per RANK (256 pooled across ranks by the gather). This
    # is the estimator's binding constraint: for P==Q the MMD^2 variance keeps a
    # 1/(m(m-1)) term from the rollout side, so lowering it raises the noise floor in
    # a way the enlarged target side cannot compensate for (n saturates by ~2048 with
    # m fixed). The band's forward cost is not the bottleneck -- these arms run ~5 h
    # against reg_ft's ~4.7 h -- so there is nothing to buy by cutting it.
    mmd_batch=64,
    # Target side enlarged to 2048 pooled samples, drawn CLASS-MATCHED from a
    # per-class FIFO (the rollout covers ~64 of 1000 classes; an unmatched target
    # reads +1.0e-3 on a perfect model and that bias does not shrink with n).
    # Costs no forward, no graph, no gradient. Measured floor sd 2.1e-4 -> 7.6e-5.
    mmd_target_n=int(os.environ.get('LAKON_MMD_TARGET_N', 2048)),
    # Confine the step's rollout to this many DISTINCT classes (d-flow's
    # classes_per_batch). Unset => the batch's own labels, which at 1000 classes give
    # ~62 distinct over 64 trajectories, i.e. ~1 trajectory per class. Set to e.g. 8
    # for 8 trajectories per class, comparable to d-flow's 6.4 on CIFAR.
    mmd_classes_per_batch=(int(os.environ['LAKON_MMD_CLASSES'])
                           if os.environ.get('LAKON_MMD_CLASSES') else None),
    mmd_target_share=_share,
))

# LR 2.5e-4, the comparison arms' rate -- NOT the 1e-5 the base config carries. The
# two original w=100 runs were trained at 1e-5, so they are NOT comparable to this
# sweep; regft_mmd_sub100 / _raw100 re-measure that weight at this LR. The control is
# therefore reg_ft @ 2.5e-4 (best FID 4.374 @ w=2.4), not regft_lr1e5 (4.402).
optimizer = {'diffusion': dict(lr=2.5e-4)}

resume_from = f'checkpoints/{name}/latest.pth'
