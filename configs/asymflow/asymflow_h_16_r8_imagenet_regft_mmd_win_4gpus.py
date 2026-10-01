"""regft + MMD on a band WINDOW sigma in [0.75, 0.85] instead of [0.875, 1.0].

Same arm as asymflow_h_16_r8_imagenet_regft_mmd_sweep_4gpus.py -- on-path FM over the
full sigma range, MMD between the rollout marginal and the noised on-path marginal,
pooled across ranks to m=256, multi-bandwidth RBF on the median heuristic -- with the
band moved DOWN and given an upper edge.

Why: with x_t = (1-s)x_0 + s*eps the data-dependent share of the variance is

    s=0.92 -> 0.75%     s=0.85 -> 3.0%     s=0.80 -> 5.9%     s=0.75 -> 10.0%

and the s*eps term is identically distributed on both sides, so it adds nothing to
the discrepancy while dominating the pairwise distances the bandwidth is set from.
Measured on synthetic data: a generated distribution 10% too narrow -- a gross
modelling error -- gives MMD^2 BELOW the noise floor at s >= 0.88 (SNR ~ 0), i.e. the
old band could not have detected it. That is the likely reason mmd_sub sat at
0.0000-0.0001 all run: not agreement, but blindness.

Cost: the trajectory above mmd_t_hi is integrated under no_grad (inference only, no
graph), so the graph spans the window alone -- 5 Heun steps / 10 evals, versus 6
steps / 12 evals before. Comparable memory and compute, ~4-13x more data signal.

    no-grad prefix : 7 Heun steps, sigma 1.00 -> 0.86   (14 evals, no graph)
    window (grad)  : sigma 0.84, 0.82, 0.80, 0.78, 0.76 (10 evals, in the graph)
"""

import os

_base_ = ['./asymflow_h_16_r8_imagenet_regft_mmd_sweep_4gpus.py']

_w = os.environ.get('LAKON_MMD_WEIGHT', '100')
_feat = os.environ.get('LAKON_MMD_FEATURE', 'subspace')
_tag = 'sub' if _feat == 'subspace' else _feat

# mmd_guidance_scale is inherited from the sweep config (LAKON_MMD_GUIDANCE); tagged
# here too, or a guided run would resume from the unguided run's checkpoint
_g = os.environ.get('LAKON_MMD_GUIDANCE', '1')
_gtag = '' if float(_g) == 1.0 else f'_g{_g}'

name = f'asymflow_h_16_r8_imagenet_regft_mmd_win_{_tag}{_w}{_gtag}_4gpus'
work_dir = f'work_dirs/{name}'

model = dict(diffusion=dict(
    mmd_weight=float(_w),
    mmd_feature=_feat,
    mmd_t_split=float(os.environ.get('LAKON_MMD_LO', 0.75)),   # window lower edge
    mmd_t_hi=float(os.environ.get('LAKON_MMD_HI', 0.85)),      # window upper edge
))

resume_from = f'checkpoints/{name}/latest.pth'
