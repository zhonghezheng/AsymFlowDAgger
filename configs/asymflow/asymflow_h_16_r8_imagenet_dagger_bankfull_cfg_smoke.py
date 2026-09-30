"""<1h smoke for the CFG arm as the 36 queued runs are configured, covering the
complement-mode code that has never executed: expert_target's `mode` parameter and
the four cm{c}{f} combinations it feeds (cmff / cmfp / cmpf / cmpp).

LAKON_CFG_CM / LAKON_FM_CM select the pair, exactly as in the real config, and the
name carries the tag so two modes can smoke concurrently without sharing a work_dir.

The band is pulled forward (mmd_start_iter=2, interval=2) so five banded iterations
land inside twelve. Everything that sets the band's COST is left at the real values
-- band_batch=4, bank_size=None, null_bank_size=2048 -- so the ~27 s/iteration bank
build is measured rather than assumed; that is the arm's known bottleneck.
"""

import os

_base_ = ['./asymflow_h_16_r8_imagenet_dagger_bankfull_cfg_4gpus.py']

_ccm = os.environ.get('LAKON_CFG_CM', 'full')
_fcm = 'full'   # pinned in the parent config; kept here so the NAME matches
_w = os.environ.get('LAKON_CFG_GAP_WEIGHT', '1')
_f = os.environ.get('LAKON_BAND_FRAC', '0.5')
_ftag = str(int(round(float(_f) * 100)))

name = ('asymflow_h_16_r8_imagenet_dagger_bankfull_'
        f'cm{_ccm[0]}{_fcm[0]}{_w}_f{_ftag}_smoke')
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
