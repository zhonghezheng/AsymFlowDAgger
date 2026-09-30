"""DAGGER-full, LARGE (capped) expert: bank_size=None (entire class per conditional
bank) + a large-but-tractable null bank. t_split=0.92 (inherited). Exact averaging
(no top-K approximation).

- bank_size=None  -> each conditional bank is the whole class (~1300 imgs, x2 with
  flips ~2600 vectors) -- the posterior support is the full class.
- null_bank_size=2048 -> ~8x the baseline (256); the null bank is drawn from the
  whole dataset, where 256 severely undersamples the 1.28M-image marginal, so this
  is the knob where a bigger bank actually improves the unconditional estimate.

Cost note: the "VAE" is a trivial affine (RGBColorEncoder), so the per-round cost is
dominated by rollout GENERATION (1024 traj x 50 NFE), which is bank-size independent.
Bank building here is n_rollout x (bank~1300 + null 2048) ~ 3.4M image loads/round --
tractable without a cache. If rounds get slow, add a preprocessed-image memmap cache.
"""

_base_ = ['./asymflow_h_16_r8_imagenet_dagger_full_4gpus.py']

name = 'asymflow_h_16_r8_imagenet_dagger_full_bankfull_4gpus'
work_dir = f'work_dirs/{name}'

model = dict(expert=dict(bank_size=None, null_bank_size=2048))

# wandb-free logging (override the full config's wandb log_config); keep the
# inherited checkpoint_config (saves iter_2500 / iter_5000) so the final model
# is available for eval.
log_config = dict(interval=100, hooks=[
    dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])

resume_from = f'checkpoints/{name}/latest.pth'
