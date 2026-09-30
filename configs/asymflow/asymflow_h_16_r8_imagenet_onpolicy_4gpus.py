"""reg_ft + fully-online on-policy band training (no replay buffer, no expert).

The DAGGER arms reach the high-sigma band through a replay buffer: a rollout round
every 500 iters fills it with expert-labelled states, and the next 500 iters
resample them. This arm removes the buffer entirely. Every iteration rolls the
CURRENT policy out from noise on the EVAL sampler/NFE grid (FlowHeunODE, 50 steps),
stops at the band edge, and trains at exactly the states that rollout just visited,
so every point is on-policy for the weights being updated -- and no expert banks are
ever built (the ~1.9h/round bank IO disappears with them).

Supervision at those on-policy states is the CFG gap

    ||v_cond - v_uncond||^2      (v = the clamp-weighted velocity u_t_pred)

evaluated per state under the row's TRUE class label and under the null label
(one batched forward of 2 x band_batch rows). It replaces DAGGER's empirical-expert
velocity target with something that needs no external target at all: it only asks
that conditioning stop changing the velocity inside the band. That band is exactly
the region inference leaves unguided (eval guidance_interval=[0, 0.88], so CFG is
off for sigma > 0.88 and the sampler uses the CONDITIONAL branch there), so driving
cond and uncond together in the band makes the model behave the same there whether
or not CFG is applied.

Band: sigma >= 0.875 -> with nfe=50 (shift=1) the visited states are
sigma = 0.98, 0.96, 0.94, 0.92, 0.90, 0.88 (6 states; sigma=1.0 is skipped, the state
there is exactly N(0,I) and carries no class information).

The optional MMD term (same one as the regft_mmd arm) is OFF here: set
LAKON_MMD_WEIGHT to enable it. NB it changes the arm's cost and gradient character
completely -- MMD needs the differentiable rollout (backprop through all 12 Heun
evals), while the CFG-gap term reads DETACHED states and needs no trajectory graph
at all. See the LR note below.

The inherited DAGGER BUFFER stays inert (no DaggerRolloutHook, so buffer_batch is
always None and nothing is ever replayed), making this reg_ft + the on-policy band
term(s). Two inherited fields are NOT inert, though: the expert is required (it
supplies v*), and t_split is set to the band edge so the on-path FM loss stops
there -- the band is supervised by the on-policy terms alone, never twice.
"""

import os

_base_ = ['./asymflow_h_16_r8_imagenet_regft_4gpus.py']

name = 'asymflow_h_16_r8_imagenet_onpolicy_4gpus'
work_dir = f'work_dirs/{name}'

warmup_iters = 500

model = dict(
    # v* is an average over real images, so the expert is REQUIRED here (unlike the
    # regft_mmd arm). Banks are deliberately much smaller than the DAGGER arms':
    # they are rebuilt every banded iteration rather than once per 500-iter round.
    expert=dict(
        # _delete_: the regft base sets expert=None, and mmcv cannot merge a dict
        # over None -- this replaces the key outright rather than updating it.
        _delete_=True,
        type='EmpiricalExpert',
        datalist_path='data/imagenet/train.txt',
        data_root='data/imagenet/train/',
        image_size=256,
        num_classes=1000,
        bank_size=16,        # real images per class (x2 with include_flips)
        null_bank_size=32,   # per-trajectory null bank (x2 flips)
        include_flips=True,
        num_workers=32,
        sample_chunk=32,
        bank_chunk=128,
    ),
    diffusion=dict(
        type='GaussianFlowOnPolicy',
        # MUST be 'full', NOT the base dagger config's 'project'. expert_target in
        # 'project' mode returns P(x_t - (1-s) x0_hat)/s_clamped - clamp_coef*x0_hat,
        # and P (proj_buffer, 768x8) keeps 8 of 768 dims -- it discards 99% of the
        # noise energy. Regressing the model's FULL-space velocity onto that target
        # leaves the whole complement as an IRREDUCIBLE residual: measured at 0.4947
        # with a PERFECT expert (x0_hat == x_0), i.e. ~15x loss_diffusion's raw 0.032
        # and nothing to do with model error. 'full' gives (x_t - x0_hat)/s_clamped,
        # which is bit-exactly the on-path FM target when x0_hat == x_0 -- so emp_fm
        # and loss_diffusion land in the same magnitude, as they should.
        # This is the known 'project diverges catastrophically' failure; the working
        # DAGGER arms (dagger_full_*) all override to 'full' for the same reason.
        complement_mode='full',
        # --- CFG-gap term ---
        # The raw gap is logged as cfg_gap (plus per-state cfg_gap_s0.980...), so the
        # true scale against loss_diffusion ~0.06 is visible from the first log line
        # -- set this from that reading rather than trusting the default.
        cfg_gap_weight=float(os.environ.get('LAKON_CFG_GAP_WEIGHT', 1.0)),
        # 'expert' = match the data-derived gap (the objective above).
        # 'zero'   = expert-free ablation driving the model's own gap to zero.
        cfg_gap_target=os.environ.get('LAKON_CFG_GAP_TARGET', 'expert'),
        # --- empirical FM term at the same on-policy states ---
        # Regresses each branch onto its own expert velocity (v_cond -> v*_cond,
        # v_uncond -> v*_uncond) -- DAGGER's loss_dagger target, taken on-policy.
        # Shares the gap term's forward and bank evaluation, so it costs only the
        # loss arithmetic. Logged as emp_fm through self.flow_loss, hence directly
        # comparable to the DAGGER arms' loss_dagger (and small, since the
        # logit-normal rescale down-weights this sigma range hard).
        # 'mass' -> p_high = P(sigma >= band_t_split), so the band term carries
        # exactly the sigma mass it has in reg_ft and matches the on-path term's
        # p_low factor. At band_t_split=0.9 that resolves to ~0.1; weighting it 1.0
        # would give sigma >= 0.9 ten times its share, and would shift the useful
        # range of w_cfg / w_align (which are read against emp_fm) by a decade.
        emp_fm_weight=(lambda v: v if v == 'mass' else float(v))(
            os.environ.get('LAKON_EMP_FM_WEIGHT', 'mass')),
        # MEMORY knob: each scored state retains its own 2 x band_batch forward
        # graph. All 6 states at band_batch=64 would hold ~768 rows of activations,
        # several times the bs=256 FM step; 2 keeps the band comparable to it.
        # Unbiased over iterations -- the states are drawn uniformly each time.
        cfg_gap_max_states=2,
        # False = the literal loss, both branches differentiated (they meet in the
        # middle). True would freeze the unconditional field and move only the
        # conditional one toward it.
        cfg_gap_detach_uncond=False,
        # null/CFG-embedding slot; matches the dataset's negative_label=1000.
        null_label=1000,
        # --- the band (shared by the CFG-gap and MMD terms) ---
        # Band lower edge. 0.9 sits INSIDE the eval CFG cutoff (guidance_interval
        # =[0, 0.88], so CFG is off above 0.88): the band covers sigma in [0.9, 1],
        # a strict subset of the unguided tail, and sigma in [0.88, 0.9] is left to
        # the on-path FM loss. With nfe=50 the visited states are
        # sigma = 0.98, 0.96, 0.94, 0.92, 0.90 -- 5 states (sigma=1.0 is skipped,
        # being exactly N(0,I) and carrying no class information).
        band_t_split=float(os.environ.get('LAKON_BAND_TSPLIT', 0.9)),
        band_nfe=50,             # match the eval sampler's NFE grid
        # rollout width AND the number of bank sets built per banded iteration --
        # the cost driver above, not just a variance knob.
        band_batch=16,
        band_sampler='FlowHeunODE',
        # --- optional MMD term (off by default) ---
        mmd_weight=float(os.environ.get('LAKON_MMD_WEIGHT', 0.0)),
        mmd_feature=os.environ.get('LAKON_MMD_FEATURE', 'subspace'),
        mmd_bandwidths=(0.25, 0.5, 1.0, 2.0, 4.0),
        mmd_unbiased=True,
        # --- shared band scheduling ---
        mmd_interval=4,                # every 4th iteration (amortises the bank IO)
        mmd_start_iter=warmup_iters,   # pure FM during the LR warmup
        mmd_eval_mode=True,            # dropout off in the rollout -> matches inference
        # must match train_cfg.grad_accum_batch_size below: the band terms run on ONE
        # micro-batch per iteration and are rescaled by this so the objective, the
        # gradient and the logs match the no-accumulation case.
        mmd_accum_steps=2,
    ),
)

# Same split as the regft_mmd arm: 2 micro-batches of 128 halves the FM step's
# activations at identical total compute and an exact gradient, leaving room for the
# band's forward graphs alongside it.
train_cfg = dict(grad_accum_batch_size=128)

# LR is left at the comparison arms' 2.5e-4 (inherited): with MMD off the CFG-gap
# gradient is an ordinary per-point regression at detached states -- no backprop
# through the solver -- so it is as benign as the FM gradient, and the arm stays
# LR-matched to reg_ft / the DAGGER arms for a like-for-like training-FID overlay.
# If you enable MMD (LAKON_MMD_WEIGHT), drop the LR to 1e-5 as regft_mmd does: that
# term DOES backprop through 12 chained network evals.

# wandb-free logging (shared fileset is full); keep the inherited EMA hook only.
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
