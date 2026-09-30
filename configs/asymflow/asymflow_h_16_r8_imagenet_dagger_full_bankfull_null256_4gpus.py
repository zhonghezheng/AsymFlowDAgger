"""DAGGER-full, full conditional bank + BASELINE null bank (256).

Same as asymflow_h_16_r8_imagenet_dagger_full_bankfull_4gpus.py (bank_size=None,
t_split=0.92, complement='full') EXCEPT null_bank_size=256 instead of 2048.
Isolates the effect of the large null bank: does bankfull's improvement come from
the full conditional bank, the 8x-larger null bank, or both?

Loads/round ~ n_rollout x (bank~1300 + null 256) ~ 1.6M (about half bankfull's 3.4M,
so ~2x faster rounds). Still conditional-loading bound; --mem=512G (host page cache).
"""

_base_ = ['./asymflow_h_16_r8_imagenet_dagger_full_4gpus.py']

name = 'asymflow_h_16_r8_imagenet_dagger_full_bankfull_null256_4gpus'
work_dir = f'work_dirs/{name}'

model = dict(expert=dict(bank_size=None, null_bank_size=256))

# wandb-free logging; keep inherited checkpoint_config (saves iter_2500 / iter_5000)
log_config = dict(interval=100, hooks=[
    dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])

resume_from = f'checkpoints/{name}/latest.pth'
