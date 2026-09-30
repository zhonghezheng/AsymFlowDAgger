"""regft_mmd, but the MMD lives in RAW latent space instead of the subspace.

Identical to asymflow_h_16_r8_imagenet_regft_mmd_4gpus.py except
mmd_feature='subspace' -> 'raw': the kernel sees the flattened 3x256x256 latent
(196608-d) rather than the AsymJiT rank-8 subspace features (2048-d). Same band
(sigma >= 0.875), same rollout, same lr. Isolates the FEATURE SPACE: does matching
the marginals in the model's own low-rank subspace (where the empirical expert
also scores) differ from matching them in the full latent?

The space not trained on is still computed under no_grad and logged, so this run
logs mmd_sub alongside mmd_raw exactly as the subspace arm does -- the two arms'
logs are directly comparable.

mmd_weight is inherited (100) but stays LAKON_MMD_WEIGHT-overridable: raw-space
MMD^2 need not have the same magnitude as subspace MMD^2, and the two arms are
only a clean space-vs-space comparison if their MMD terms enter the loss at a
comparable scale.
"""

_base_ = ['./asymflow_h_16_r8_imagenet_regft_mmd_4gpus.py']

name = 'asymflow_h_16_r8_imagenet_regft_mmd_raw_4gpus'
work_dir = f'work_dirs/{name}'

model = dict(diffusion=dict(mmd_feature='raw'))

resume_from = f'checkpoints/{name}/latest.pth'
