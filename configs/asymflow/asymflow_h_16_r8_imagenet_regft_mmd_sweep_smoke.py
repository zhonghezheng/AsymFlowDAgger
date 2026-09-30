"""<1h smoke for the MMD arm AS CURRENTLY CONFIGURED (the sweep config, lr 2.5e-4,
mmd_target_n=2048, class-matched FIFO), after the two bugs that made this path
unreachable:

  1. the _mmd_active latch, which burned on the warmup check so _mmd_loss never ran;
  2. the NameError on `labels` in _target_feats, which only became reachable once (1)
     was fixed.

Both were invisible because the code below mmd_start_iter is the only part that had
ever executed. So this smoke is built to reach the parts that had not:

  mmd_start_iter=4 (NOT 0) -- iters 0-3 warm the per-class FIFO, so from iter 4 the
  CLASS-MATCHED branch of _target_feats is taken. At mmd_start_iter=0 the cache is
  empty, `not self._feat_cache` short-circuits to the on-path fallback, and the
  branch that crashed is skipped -- the smoke would pass while proving nothing.

Run on >=2 ranks: _gather_feats returns early when world_size==1, so a 1-GPU smoke
leaves the all_gather and the *world_size loss scaling untested.
"""

_base_ = ['./asymflow_h_16_r8_imagenet_regft_mmd_sweep_4gpus.py']

name = 'asymflow_h_16_r8_imagenet_regft_mmd_sweep_smoke'
work_dir = f'work_dirs/{name}'

total_iters = 12

model = dict(diffusion=dict(
    mmd_start_iter=4,   # 4 warmup iters fill the FIFO, then 8 real MMD iters
    mmd_interval=1,
))

# log every iteration: train_grad_accum only populates log_vars on logged steps, so
# at the default interval a 12-iteration smoke would report mmd values once.
train_cfg = dict(grad_accum_batch_size=128, log_interval=1)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])

data = dict(workers_per_gpu=4, prefetch_factor=2)

checkpoint_config = dict(interval=100000, by_epoch=False, out_dir='checkpoints/')
evaluation = []
workflow = [('train', total_iters)]
resume_from = None
