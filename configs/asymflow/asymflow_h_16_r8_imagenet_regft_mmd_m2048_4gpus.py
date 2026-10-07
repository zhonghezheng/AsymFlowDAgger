"""regft + raw-space RBF MMD on the band sigma >= 0.88, at 2048 pooled rollouts against
10240 pooled targets, on 4xH200.

The unguided control raw100_cls8_shboth_4gpus
(asymflow_h_16_r8_imagenet_regft_mmd_sweep_4gpus.py with LAKON_MMD_FEATURE=raw,
LAKON_MMD_CLASSES=8, LAKON_MMD_SHARE=both) scaled up on both sides of the estimator:

                      control                     this arm
  rollouts (pooled)   256  = 64/rank x 4          2048 = 512/rank x 4
  targets  (pooled)   2048 (8 per rollout)        10240 (5 per rollout)
  classes             8/rank, 8 traj/class        64/rank, 8 traj/class (256 pooled)
  distances           TF32                        full fp32

Everything else is the control's: band sigma = 0.98 .. 0.88 (mmd_t_split=0.875, 6 Heun
steps, gradient from sigma=1), unguided rollout, Gaussian/RBF kernel with 'mean' widths
(0.25 .. 4), mmd_target_share='both', weight 100, LR 2.5e-4, 5000 iters, MMD from iter
500, full-range on-path flow matching.

Why the sizes: raw at m=256 sat at its noise floor (mmd_raw read +-0.0000 all run;
tools/tsplit_snr_m.py put it at z~0.17 at t_split=0.875). For P==Q the unbiased
estimator's sd falls like 1/m, so 8x the rollouts lowers the floor ~8x; the target
side is raised so it does not become the binding term.

How it runs: under mmd_chunk (GaussianFlowMMD._mmd_chunked_step). The band is rolled
out ONCE for all 512 trajectories with its graph kept, the pooled MMD^2 and its exact
gradient w.r.t. every band state are taken on detached copies (kernel rows sharded
across ranks, mmd2_rbf_sharded), and those state gradients are backpropagated in one
pass. It runs on the first FM micro-batch (under DDP no_sync, which the in-forward
backward needs) BEFORE the FM forward, so the band graph never coexists with the FM
graph. Memory: the band graph measured ~5 GB per 64 trajectories (peak 74.8 -> 79.8 GB
when MMD switched on in raw100_cls8_shboth), so ~40 GB at 512 on a 141 GB H200.
If that ever OOMs, LAKON_MMD_CHUNK=128 rolls out without a graph and REPLAYS the band
in chunks of 128 (same gradient, one extra no-grad rollout, ~20% more band compute;
check mmd_replay_err ~ 0 in the log then).

Every knob the sweep config reads from LAKON_* env vars is pinned here, so stray env
vars cannot change the arm. Only the weight stays overridable, and it is in the name.
LAKON_MMD_CHUNK picks single pass vs replay (memory only, not the objective); LAKON_U8_CACHE
picks the target image cache (inherited; della needs it set).
"""

import os

_base_ = ['./asymflow_h_16_r8_imagenet_regft_mmd_sweep_4gpus.py']

_w = os.environ.get('LAKON_MMD_WEIGHT', '100')
name = f'asymflow_h_16_r8_imagenet_regft_mmd_raw{_w}_cls8_shboth_m2048_n10240_4gpus'
work_dir = f'work_dirs/{name}'

_m = 512   # rollouts per rank; x 4 ranks = 2048 pooled

model = dict(diffusion=dict(
    mmd_weight=float(_w),
    mmd_feature='raw',
    mmd_t_split=0.875,
    mmd_t_hi=None,
    mmd_guidance_scale=1.0,
    mmd_batch=_m,
    # >= mmd_batch -> single pass; smaller -> chunked replay (memory fallback)
    mmd_chunk=int(os.environ.get('LAKON_MMD_CHUNK', _m)),
    # pooled; 10240 / 4 ranks / 512 rollouts = 5 class-matched targets per rollout
    mmd_target_n=10240,
    # PER RANK: 64 classes x 8 trajectories each, as cls8 had 8 per class
    mmd_classes_per_batch=_m // 8,
    mmd_target_share='both',
    # inherited: mmd_accum_steps=2 == 256/rank / grad_accum_batch_size 128, which
    # mmd_chunk also needs (>= 2 micro-batches, so the MMD's micro-batch is no_sync)
))

# ~45 s/iter once the MMD starts: the inherited save interval (2500) would put ~31 h of
# work between checkpoints. Keep only the latest (+ the must-save final one).
checkpoint_config = dict(interval=500)

resume_from = f'checkpoints/{name}/latest.pth'
