"""DAGGER full, entire-class disk bank, t_split=0.75 (mid-sigma expert).

Same as asymflow_h_16_r8_imagenet_dagger_full_4gpus.py (full complement, entire-class
disk bank, proportional weighting, label-time dropout, proportional classes) EXCEPT
the timestep split moves from 0.88 to 0.75, so the expert/rollout covers a broader,
more informative mid-sigma band (sigma >= 0.75) where the large bank reconstructs real
data rather than collapsing to the class mean.

Everything keys off t_split automatically:
  - diffusion.t_split=0.75 -> high_sigma_fraction() recomputes p_high=P(sigma>=0.75)
    -> minibatch carve n_roll=round(bs*p_high), n_on=bs-n_roll
    -> roll_weight='proportional' convex weight w = n_roll/bs (stays aligned).
  - DaggerRolloutHook t_split omitted -> inherits diffusion.t_split=0.75 (captures
    sigma >= 0.75).
Eval CFG is unchanged: guidance_interval stays [0, 0.88] (separate from t_split).
Distinct name -> separate checkpoints / work_dir / wandb run.
"""

_base_ = ['./asymflow_h_16_r8_imagenet_dagger_full_4gpus.py']

name = 'asymflow_h_16_r8_imagenet_dagger_full_ts075_4gpus'
work_dir = f'work_dirs/{name}'

model = dict(diffusion=dict(t_split=0.75))

# separate wandb run (same project) for side-by-side comparison
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(project='asymflow-dagger', name=name, mode='offline')),
    ])

resume_from = f'checkpoints/{name}/latest.pth'
