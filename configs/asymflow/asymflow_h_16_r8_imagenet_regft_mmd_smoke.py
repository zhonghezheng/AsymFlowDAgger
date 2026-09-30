"""Smoke test for the regft_mmd arm: does the differentiable band rollout + MMD
run, and does it FIT in memory next to the bs=256 flow-matching step?

Real memory profile (samples_per_gpu=256, compile ON, checkpointing ON) but only a
handful of iterations and no eval, so it can run on 1 GPU. The MMD branch is on
from iter 0 (mmd_start_iter=0) and every iteration logs mmd_sub / mmd_raw plus the
per-step values, so this also reveals the raw MMD scale relative to loss_diffusion
-- which is what mmd_weight has to be set against.

Override the rollout batch with LAKON_MMD_BATCH (the memory knob) and the trained
space with LAKON_MMD_FEATURE. Launch with --no-validate.
"""

import os

_base_ = ['./asymflow_h_16_r8_imagenet_regft_mmd_4gpus.py']

name = 'asymflow_h_16_r8_imagenet_regft_mmd_smoke'
work_dir = f'work_dirs/{name}'

total_iters = 8

model = dict(
    diffusion=dict(
        mmd_start_iter=0,     # exercise the MMD branch immediately
        mmd_batch=int(os.environ.get('LAKON_MMD_BATCH', 64)),
        mmd_feature=os.environ.get('LAKON_MMD_FEATURE', 'both'),  # log + train both spaces
        # The gradient probe is OFF: it needs extra retain_graph backward passes
        # through the band graph, which the step-checkpointed rollout cannot serve
        # (recomputed tensors are freed after the first backward), and it was what
        # tipped the first smoke into OOM. mmd_weight is being set by hand instead;
        # the logged mmd_sub / mmd_raw values give the loss-scale half of the picture.
        mmd_grad_probe=False,
    ),
)

# inherited from the real config, restated here so the smoke measures the real
# memory profile: 2 micro-batches of 128 (see that config's note). log_interval=1
# (vs 10) because train_grad_accum only POPULATES log_vars on logged steps, so with
# the default an 8-iteration smoke would report mmd values for iter 0 alone.
train_cfg = dict(grad_accum_batch_size=128, log_interval=1)

data = dict(workers_per_gpu=4, prefetch_factor=2)

log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])

checkpoint_config = dict(interval=100000, by_epoch=False, out_dir='checkpoints/')
evaluation = []
workflow = [('train', total_iters)]
resume_from = None
