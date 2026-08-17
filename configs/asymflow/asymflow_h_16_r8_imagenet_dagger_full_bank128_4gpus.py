"""DAGGER full bank_size sweep: bank_size=128 (x2 with include_flips), t_split=0.88.

Sweeps ONLY the conditional bank size vs asymflow_h_16_r8_imagenet_dagger_full_4gpus.py
(complement=full, t_split=0.88, bs=256, lr=2.5e-4, warmup 500, flips, frac_on_path=0.5).
Distinct name -> separate checkpoints / work_dir / wandb run.
"""

_base_ = ['./asymflow_h_16_r8_imagenet_dagger_full_4gpus.py']

name = 'asymflow_h_16_r8_imagenet_dagger_full_bank128_4gpus'
work_dir = f'work_dirs/{name}'

model = dict(expert=dict(bank_size=128))

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(project='asymflow-dagger', name=name, mode='offline')),
    ])

resume_from = f'checkpoints/{name}/latest.pth'
