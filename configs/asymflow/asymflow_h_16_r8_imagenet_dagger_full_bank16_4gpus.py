"""DAGGER full bank_size sweep: bank_size=16 (x2 with include_flips), t_split=0.92.

Sweeps ONLY the conditional bank size vs asymflow_h_16_r8_imagenet_dagger_full_4gpus.py.
No wandb + no checkpoint saves (shared fileset is full; the sweep only needs the FID
curves, which go to the text log). Distinct name -> separate work_dir.
"""

_base_ = ['./asymflow_h_16_r8_imagenet_dagger_full_4gpus.py']

name = 'asymflow_h_16_r8_imagenet_dagger_full_bank16_4gpus_split95'
work_dir = f'work_dirs/{name}'

model = dict(expert=dict(bank_size=16))

# don't save checkpoints: fileset full -> avoid EDQUOT on the 7.6 GB writes; the
# sweep only needs the logged FID curves (interval > total_iters => never saves).
checkpoint_config = dict(interval=1000000, must_save_interval=1000000,
                         by_epoch=False, max_keep_ckpts=1, out_dir='checkpoints/')

# wandb-free logging (override the full config's wandb log_config)
log_config = dict(interval=100, hooks=[
    dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])

resume_from = None
