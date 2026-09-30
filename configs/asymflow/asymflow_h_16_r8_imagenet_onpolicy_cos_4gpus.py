"""on-policy band arm sweeping the ALIGNMENT weight (cosine) instead of the CFG gap.

Identical to asymflow_h_16_r8_imagenet_onpolicy_wcfg_4gpus.py in every other
respect -- same conditional-only rollout, same band, same banks, same per-point CFG
dropout on the empirical-FM term -- except the band's second term. Writing
a = v_cond - v*_cond and b = v_uncond - v*_uncond:

    wcfg arm : w_cfg  * || a - b ||^2                      (direction AND magnitude)
    this arm : w_align * ( - mean_rows <a/||a||, b/||b||> ) (direction ALONE)

||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b> ties the two branches' residual sizes into
the same term that couples their directions. The cosine strips the magnitudes out:
it is scale free, bounded in [-w, w], and therefore cannot be lowered by making both
branches enormously wrong in a shared direction -- the failure mode a raw <a,b>
reward has. Residual magnitudes stay the business of emp_fm and the on-path FM loss.

cfg_gap_weight is forced to 0 here, so the sweep is over the alignment term alone.

Weights are NOT comparable with the wcfg arm's: ||a - b||^2 carries the residual
scale while the cosine is O(1) and contributes at most w in absolute value. Hence
the much smaller sweep range -- read the logged cfg_cos (the raw cosine, unweighted)
against emp_fm and loss_diffusion (~0.06) to judge whether a weight bites. If
cfg_cos sits at ~0 the two residuals are uncorrelated and no weight will matter.
"""

import os

_base_ = ['./asymflow_h_16_r8_imagenet_onpolicy_wcfg_4gpus.py']

_w = os.environ.get('LAKON_CFG_ALIGN_WEIGHT', '0.01')

name = f'asymflow_h_16_r8_imagenet_onpolicy_cos{_w}_4gpus'
work_dir = f'work_dirs/{name}'

model = dict(
    diffusion=dict(
        cfg_align_weight=float(_w),
        cfg_gap_weight=0.0,   # alignment INSTEAD of the full CFG term
    ),
)

resume_from = f'checkpoints/{name}/latest.pth'
