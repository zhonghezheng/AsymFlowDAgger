"""Smoke test for the onpolicy arm: does the online band rollout + CFG-gap term run,
and does it FIT next to the bs=256 flow-matching step?

Real memory profile (samples_per_gpu=256, compile ON, checkpointing ON) but only a
handful of iterations and no eval, so it runs on 1 GPU. The band is on from iter 0
and every iteration logs cfg_gap plus the per-state values, so this also reveals the
raw gap scale relative to loss_diffusion -- which is what cfg_gap_weight has to be
set against. Launch with --no-validate.

LAKON_CFG_GAP_STATES caps the scored states (the memory knob); set it to 6 to price
the whole band. LAKON_MMD_WEIGHT additionally switches the MMD term on, which makes
the rollout differentiable -- expect a large jump in memory.
"""

import os

_base_ = ['./asymflow_h_16_r8_imagenet_onpolicy_4gpus.py']

name = 'asymflow_h_16_r8_imagenet_onpolicy_smoke'
work_dir = f'work_dirs/{name}'

total_iters = 8

model = dict(
    diffusion=dict(
        mmd_start_iter=0,   # exercise the band immediately
        cfg_gap_max_states=int(os.environ.get('LAKON_CFG_GAP_STATES', 2)),
        band_batch=int(os.environ.get('LAKON_BAND_BATCH', 16)),
    ),
)

# log_interval=1 because train_grad_accum only POPULATES log_vars on logged steps.
train_cfg = dict(grad_accum_batch_size=128, log_interval=1)

data = dict(workers_per_gpu=4, prefetch_factor=2)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
checkpoint_config = dict(interval=100000, by_epoch=False, out_dir='checkpoints/')
evaluation = []
workflow = [('train', total_iters)]
resume_from = None
