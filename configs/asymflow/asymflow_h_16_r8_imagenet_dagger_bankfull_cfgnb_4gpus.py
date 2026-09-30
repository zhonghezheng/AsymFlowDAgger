"""VARIANT of the online-DAGGER + CFG arm: the null bank drawn from the BATCH's classes.

Identical to asymflow_h_16_r8_imagenet_dagger_bankfull_cfg_4gpus.py (frac_on_path=0.5,
fully online, conditional-only rollouts, CFG dropout applied at label time inside
emp_fm) except band_null_from_batch=True: x0_hat_uncond is estimated from a bank drawn
from the ~64 classes the rollout used, instead of the whole 1.28M-image pool. Applies
to BOTH band terms, since they share one build_banks call.

Run as a variant so the choice is measured rather than assumed. The standard arms
(dagger_bankfull_cfg*_f50) are the control -- same weights, same everything else.

The prior is that this HURTS, and the reason is worth stating so the comparison is
read correctly: v*_uncond is the target for the model's unconditional branch, whose
null embedding is trained on the FULL marginal. Restricting the bank makes target and
prediction refer to different quantities -- "class c vs this batch's 64-class mixture"
rather than "class c vs the data". Simulated at the near-uniform weights that hold at
sigma>=0.92: error on Delta = x0_cond - x0_uncond rises 0.61 -> 1.31 (SNR 15.8 -> 7.3),
and the reference moves every iteration with the class draw rather than averaging out.

If it nonetheless helps, that would say the CFG gap benefits from a LOCAL contrast
(class vs nearby classes) rather than a global one -- which would be a real finding
about what the term should be measuring.
"""

import os

_base_ = ['./asymflow_h_16_r8_imagenet_dagger_bankfull_cfg_4gpus.py']

_w = os.environ.get('LAKON_CFG_GAP_WEIGHT', '0')
_f = os.environ.get('LAKON_BAND_FRAC', '0.5')
_ftag = str(int(round(float(_f) * 100)))

_proj = os.environ.get('LAKON_CFG_PROJECT', '0') == '1'
_ptag = 'p' if _proj else ''

name = f'asymflow_h_16_r8_imagenet_dagger_bankfull_cfgnb{_ptag}{_w}_f{_ftag}_4gpus'
work_dir = f'work_dirs/{name}'

model = dict(diffusion=dict(band_null_from_batch=True, cfg_gap_project=_proj))

resume_from = f'checkpoints/{name}/latest.pth'
