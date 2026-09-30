"""reg_ft + per-NFE-step MMD (distribution matching) in the high-sigma band.

Same objective as asymflow_h_16_r8_imagenet_regft_4gpus.py (full-range on-path
flow matching, no empirical expert, no rollout hook) PLUS a distribution-level
term: every iteration the current policy is rolled out from noise on the EVAL
sampler/NFE grid (FlowHeunODE, 50 steps) with gradient, and at each visited state
with sigma >= mmd_t_split an MMD is taken between

    the ROLLOUT marginal          {x_sigma}                          and
    the NOISED ON-PATH marginal   {(1-sigma) x_0 + sigma eps}

using the same minibatch rows (hence an identical class mixture) on both sides.
Backprop runs through every Heun step in the band, so the term trains the policy's
own high-sigma marginals rather than a per-point velocity target -- the
distribution-level counterpart to the DAGGER arms' empirical-expert velocity.

Band: sigma >= 0.875 -> with nfe=50 (shift=1) the visited states are
sigma = 0.98, 0.96, 0.94, 0.92, 0.90, 0.88 (6 MMD terms; sigma=1.0 is skipped, both
sides are exactly N(0,I) there). 6 Heun steps = 12 network evals at mmd_batch=64,
with gradient -- roughly 3 extra bs-256 forwards per iteration. 0.875 sits just
below the eval CFG cutoff (guidance_interval=[0, 0.88]), so the band covers exactly
the unguided tail of inference; the rollout here is unguided too.

The inherited DAGGER stream stays inert (expert=None, no DaggerRolloutHook), so
this is exactly reg_ft + MMD.
"""

import os

_base_ = ['./asymflow_h_16_r8_imagenet_regft_4gpus.py']

name = 'asymflow_h_16_r8_imagenet_regft_mmd_4gpus'
work_dir = f'work_dirs/{name}'

warmup_iters = 500

model = dict(
    diffusion=dict(
        type='GaussianFlowMMD',
        # --- MMD term ---
        # coefficient on mean_over_band_steps(MMD^2). The unbiased MMD^2 runs ~1e-3
        # at mmd_batch=64 (its noise floor between identical distributions is
        # ~2e-4), against loss_diffusion ~0.06 -- so 1e2 puts the MMD term at order
        # 0.1, a couple of times the flow-matching loss: a strong but not
        # overwhelming distribution term. The raw (unweighted) MMD is logged as
        # mmd_sub / mmd_raw plus per-step mmd_sub_s0.980..., so the true scale is
        # visible from the first log line.
        mmd_weight=float(os.environ.get('LAKON_MMD_WEIGHT', 100.0)),
        # band lower edge. INDEPENDENT of diffusion.t_split (inherited None from
        # regft), so the flow-matching loss stays full-range -- t_split only ever
        # carves the FM/DAGGER streams, mmd_t_split only defines the MMD band.
        mmd_t_split=0.875,
        mmd_nfe=50,              # match the eval sampler's NFE grid
        mmd_sampler='FlowHeunODE',
        mmd_batch=64,            # rollout == on-path sample count per step
        # 'subspace' = MMD on feat_fn(x) (AsymJiT rank-8 subspace, 2048-d; the space
        # the empirical expert scores in). 'raw' = flattened latents (196608-d).
        # 'both' trains on the sum. The space(s) not trained on are still computed
        # under no_grad and logged, so mmd_sub and mmd_raw are always both visible.
        mmd_feature=os.environ.get('LAKON_MMD_FEATURE', 'subspace'),
        mmd_bandwidths=(0.25, 0.5, 1.0, 2.0, 4.0),  # multipliers on the median heuristic
        mmd_unbiased=True,       # diagonal-free estimator (can go slightly negative)
        mmd_interval=1,          # every iteration
        mmd_start_iter=warmup_iters,  # pure FM during the LR warmup, like the DAGGER rounds
        mmd_eval_mode=True,      # dropout off inside the rollout -> matches inference
        # must match train_cfg.grad_accum_batch_size below: the MMD runs on ONE
        # micro-batch per iteration and is rescaled by this so the objective, the
        # gradient and the logs are identical to the no-accumulation case.
        mmd_accum_steps=2,
    ),
)

# The bs=256 flow-matching graph (~115 GB of the 140 GB card) leaves no room for the
# differentiable band graph (~19 GB at mmd_batch=64) -- the first attempt OOMed at
# 137.7 GB before backward even started. Splitting the FM step into 2 micro-batches
# of 128 halves its activations (~57 GB) at identical total compute and an exact
# gradient (train_grad_accum accumulates then averages), which buys far more room
# than the band graph needs. Preferred over checkpointing the band evals: the net
# already checkpoints per block, and nesting non-reentrant checkpoints inside that
# breaks (see mmd_step_checkpoint).
train_cfg = dict(grad_accum_batch_size=128)

# The MMD term backprops through 12 chained network evals, so its gradient is far
# less benign than the per-point FM gradient: drop the LR from the comparison
# arms' 2.5e-4 to 1e-5. NB this makes the arm not LR-matched to reg_ft / the DAGGER
# arms, so its training-FID curve is not a like-for-like overlay with theirs.
optimizer = {'diffusion': dict(lr=1e-5)}

# wandb-free logging (shared fileset is full); keep the inherited EMA hook only --
# the regft config's WandbTrajectoryHook needs wandb and draws extra samples.
custom_hooks = [
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('diffusion_ema', ),
        interp_mode='lerp',
        interval=1,
        start_iter=0,
        momentum_policy='fixed',
        interp_cfg=dict(momentum=0.9999),
        priority='VERY_HIGH'),
]

log_config = dict(interval=100, hooks=[
    dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])

resume_from = f'checkpoints/{name}/latest.pth'
