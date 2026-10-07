"""<1h smoke for the CFG arm as the 36 queued runs are configured, covering the
complement-mode code that has never executed: expert_target's `mode` parameter and
the four cm{c}{f} combinations it feeds (cmff / cmfp / cmpf / cmpp).

The CFG gap is now always 'full' (no LAKON_CFG_CM); the name keeps the cm tag so it
matches the real config's.

The band is pulled forward (mmd_start_iter=2, interval=2) so five banded iterations
land inside twelve. Everything that sets the band's COST is left at the real values
-- band_rows 'auto', bank_size=None, null_bank_size=2048 -- so the ~27 s/iteration bank
build is measured rather than assumed; that is the arm's known bottleneck.
"""

import os

_base_ = ['./asymflow_h_16_r8_imagenet_dagger_bankfull_cfg_4gpus.py']

_ccm = 'full'   # the CFG gap is always 'full'; kept here so the NAME matches
_fcm = 'full'   # pinned in the parent config; kept here so the NAME matches
_w = os.environ.get('LAKON_CFG_GAP_WEIGHT', '1')
_f = os.environ.get('LAKON_BAND_FRAC', '0.5')
_ftag = str(int(round(float(_f) * 100)))
# LAKON_BAND_SOURCE=onpath is the parent's regft_emp arm; named apart so the two arms
# can smoke concurrently without sharing a work_dir
_bsrc = os.environ.get('LAKON_BAND_SOURCE', 'rollout')

name = ('asymflow_h_16_r8_imagenet_'
        + ('regft_emp_' if _bsrc == 'onpath' else 'dagger_bankfull_')
        + f'cm{_ccm[0]}{_fcm[0]}{_w}_f{_ftag}_smoke')
work_dir = f'work_dirs/{name}'

total_iters = 12

model = dict(diffusion=dict(
    mmd_start_iter=2,
    mmd_interval=2,     # banded iters 2, 4, 6, 8, 10
))

log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
data = dict(workers_per_gpu=4, prefetch_factor=2)

checkpoint_config = dict(interval=100000, by_epoch=False, out_dir='checkpoints/')
evaluation = []
workflow = [('train', total_iters)]
resume_from = None
