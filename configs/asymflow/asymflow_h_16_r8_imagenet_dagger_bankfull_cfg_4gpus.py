"""Fully ONLINE DAGGER-bankfull + a w_cfg CFG-gap term. No replay buffer anywhere.

Base: asymflow_h_16_r8_imagenet_dagger_full_bankfull_4gpus.py for the expert
(bank_size=None, null_bank_size=2048) and the schedule. The DAGGER OBJECTIVE is
reproduced, but every rollout point is generated fresh with the CURRENT weights each
banded iteration instead of being replayed from a buffer up to 500 iterations stale:

    loss = w_on   * mean_onpath [ sigma < 0.92, PLUS a real-data mix-in inside the band ]
         + w_emp  * mean_online_rollout [ v* target, one branch per point, 10% dropout ]
         + w_cfg  * mean_online_rollout || (v_cond - v_uncond) - (v*_cond - v*_uncond) ||^2

Mass matching (band_frac_on_path=0.5, DAGGER's frac_on_path), with p_high = P(sigma
>= 0.92) = 0.08 measured from the model's own timestep sampler:

    w_on  = p_low + p_high*f = 0.96      <- 4.17% of on-path rows drawn from the band
    w_emp = p_high*(1 - f)   = 0.04
    ------------------------------------
    total                    = 1.00      <- exactly reg_ft's sigma allocation

so half the band keeps a GROUND-TRUTH anchor and the expert owns only the other half.
The previous buffer-free arms had no such anchor (frac_on_path was dead config), which
left v* -- a smoothed bank average -- supervising the band alone.

w_cfg is added OUTSIDE that normalisation with its own weight, so w_cfg=0 is the
control: online DAGGER with no CFG term.

COST: no buffer means banks are rebuilt every banded iteration rather than once per
500-iteration round -- band_batch * (~1300 + 2048) images at ~500 img/s. Budget with
band_batch (default 'auto', = 5 here) and LAKON_BAND_INTERVAL (default 1).
"""

import os

_base_ = ['./asymflow_h_16_r8_imagenet_dagger_full_bankfull_4gpus.py']

_w = os.environ.get('LAKON_CFG_GAP_WEIGHT', '0')
# band_frac_on_path: share of the band's sigma mass kept on REAL data (true velocity)
# instead of the expert. 0.5 = DAGGER's setting; 0.0 = expert owns the band outright,
# which is what every earlier buffer-free arm silently did.
_f = os.environ.get('LAKON_BAND_FRAC', '0.5')
_ftag = str(int(round(float(_f) * 100)))

# complement treatment, INDEPENDENTLY per band term. Both terms always keep both
# halves (subspace + complement); these choose how the complement is computed:
#   'full'    -> P_perp((x_t - x0_hat)/sigma_c), the head's native form
#   'project' -> -clamp_coef * x0_hat_comp, the low-rank prior
# The subspace half is identical either way (verified to 1.7e-5).
# Complement treatment for BOTH band terms, PINNED to 'full'. This is not a tuning
# choice -- it is the only one in the right space.
#
# AsymFlow (arXiv:2605.12964) defines u_A := P.eps - x_0 (Eq. 3) as the target for the
# NETWORK'S RAW OUTPUT, assembles the full velocity with
#     u = P u_A + (I - P)(x_t + u_A) / sigma_t                                 (Eq. 5)
# and takes the MSE on the ASSEMBLED velocity (Eq. 2). AsymJiT does that assembly
# inside its own forward, so self.pred() returns u, never u_A -- which means every
# quantity the band compares (v_cond, v_uncond, the sampler's rollout states) is
# assembled. The target must live in the same space.
#
#   expert_target('full')          = (x_t - x0_hat)/sigma_c
#   asymflow_velocity(u_A_emp)     = (x_t - x0_hat)/sigma_c   <- verified to 1.8e-15
#
# so 'full' IS the assembled image of the derived empirical asym target. 'project'
# leaves u_A_emp in raw-head space and compares it against an assembled prediction;
# the error is exactly the x_t_comp/sigma_c term the head injects. Measured: emp_fm
# 0.033 -> 0.390 (11.7x) with emp_fm_subfrac inverting 0.831 -> 0.057. The CFG gap
# looked merely neutral only because that term is common to both branches and cancels
# in the difference (cmff vs cmpf agreed to <=0.05 FID at every weight).
#
# Verified prerequisites for the equivalence: scale_buffer = 1.0 -> sk = 1 at all
# sigma, and proj_buffer orthonormal to 3.2e-5.
_ccm = 'full'      # CFG gap        -- do not project
_fcm = 'full'      # emp_fm         -- do not project
_cmtag = f'cm{_ccm[0]}{_fcm[0]}'                   # always 'cmff'

# Softmax temperature on the NULL bank only (d2 -> d2/T). T=1 is exact Bayes over
# the empirical prior, which in 2048-d saturates to one image -- measured ESS ~1.1 of
# 2048 at sigma=0.92, i.e. v*_uncond is a nearest-neighbour lookup, not a mean.
# T=100 measures ESS ~1022 with ||v*_c - v*_u|| at 0.94x its T=1 value. The
# uncond_bank SAMPLING is unchanged: still an independent null_bank_size=2048 draw
# from the whole dataset per trajectory.
_nt = os.environ.get('LAKON_NULL_TEMP', '1')
# ADAPTIVE temperature: rescale d2 so its post-scaling sd equals this, per row and
# per bank. Tracks sigma automatically -- the raw spread runs 10.2 at sigma=0.98 to
# 287 at 0.88, so a FIXED temperature (LAKON_NULL_TEMP) over-smooths the high-sigma
# end. LAKON_TEMP_SCOPE=null applies it to the unconditional bank only; 'both' also
# smooths v*_cond, making it a class-restricted posterior MEAN rather than a
# nearest-neighbour lookup.
_ts = os.environ.get('LAKON_TEMP_SPREAD')
_tscope = os.environ.get('LAKON_TEMP_SCOPE', 'both')

_ncls = os.environ.get('LAKON_BAND_NCLS')
_nb = os.environ.get('LAKON_BAND_NULLBATCH', '0')
_clstag = ((f'_c{_ncls}' if _ncls else '') + ('nb' if _nb != '0' else '')
           + ('' if _nt == '1' else f'_T{_nt}')
           + ('' if not _ts else f'_ts{_ts}{"u" if _tscope == "null" else "b"}'))
# band interval: tagged only when off the default 8, so existing run names (and their
# resume checkpoints) are unchanged while other intervals get their own work_dir.
# band interval. DEFAULT 1: the band fires every iteration. Interval 8 applied the
# on-policy terms on 1/8 of the optimizer steps with no compensating rescale (the band
# loss is added as w * term * acc, acc = mmd_accum_steps -- nothing divides by the
# interval), so it was a different, much weaker objective, not a cheaper estimate of
# the same one. Every cmff result before 2026-09-28 was trained that way.
#
# The TAG RULE deliberately still keys on 8, not on the default: a default run
# resolves to '..._i1_...', so it can never collide with (or resume from) the
# untagged interval-8 checkpoints already on disk, and arms queued with an explicit
# LAKON_BAND_INTERVAL=1 keep exactly the names their chained sweeps point at.
_bint = int(os.environ.get('LAKON_BAND_INTERVAL', 1))
_clstag += '' if _bint == 8 else f'_i{_bint}'
# uncond (null) bank size. Measured ESS of the x0_hat softmax is ~1-5 across the whole
# band and FLAT in M (tools/null_bank_ess.py: at sigma=0.92, ESS 1.0 at M=512 vs 1.9 at
# M=2048, and the sigma=0.98 column runs 3.0/2.1/5.8/5.1/2.7/4.6/2.4/5.0/9.9 over
# M=256..65536 -- non-monotone, i.e. noise). The softmax is a hard argmin in 2048
# feature dims, so a bigger null bank buys effectively nothing while costing
# band_traj * null_bank_size image loads every banded iteration -- 61% of the band's
# IO. Default stays 2048 so existing runs and the f0/f50 pairing are untouched.
_ubk = int(os.environ.get('LAKON_NULL_BANK', 2048))
_clstag += '' if _ubk == 2048 else f'_ub{_ubk}'
# space the expert's posterior WEIGHTS are scored in: 'feat' (rank-8 subspace, the
# historical default) or 'latent' (the full on-path diffusion state). Tagged only when
# off the default, so every existing run name is unchanged.
_kspace = os.environ.get('LAKON_KERNEL_SPACE', 'feat')
_clstag += '' if _kspace == 'feat' else '_klat'
# per-GPU batch. Default 256 = the 4xH200 layout every run so far used (global 1024).
# For 8 GPUs set 128: global batch stays 1024, per-GPU activations halve (the 256
# layout peaks ~130 GB, over an 80 GB H100), no grad accumulation is needed so
# mmd_accum_steps=1 stays correct, and the band's trajectory count -- round(p_high *
# per-GPU batch) -- stays matched in total (8 x ~10-11 vs 4 x 21). Tagged when off
# default so an 8-GPU run can never share a name with its 4-GPU counterpart.
_spg = int(os.environ.get('LAKON_SAMPLES_PER_GPU', 256))
_clstag += '' if _spg == 256 else f'_bs{_spg}'
name = ('asymflow_h_16_r8_imagenet_dagger_bankfull_'
        f'{_cmtag}{_w}_f{_ftag}{_clstag}_4gpus')
work_dir = f'work_dirs/{name}'

model = dict(
    expert=dict(null_temp=float(_nt),
                null_bank_size=_ubk,
                kernel_space=_kspace,
                temp_spread=(float(_ts) if _ts else None),
                temp_spread_null_only=(_tscope == 'null')),
    diffusion=dict(
    type='GaussianFlowOnPolicy',
    # --- the online CFG term ---
    cfg_gap_weight=float(_w),
    cfg_complement_mode=_ccm,
    emp_fm_complement_mode=_fcm,
    cfg_gap_target='expert',
    cfg_align_weight=0.0,
    emp_fm_weight='mass',     # p_high*(1-f): the band's share, minus real data's half
    mmd_weight=0.0,
    allow_no_band_term=True,  # lets the w_cfg=0 control run through the same path
    onpath_truncate='auto',   # on-path FM stops at the band edge (emp_fm owns it)
    onpath_truncate_rescale=True,
    band_frac_on_path=float(_f),
    null_label=1000,
    # --- the online band: matches DAGGER's t_split so both cover the same sigmas ---
    band_t_split=0.92,
    band_nfe=50,
    # 'auto' -> round(p_high * batch / n_band_states): the band supplies the same
    # point budget the DAGGER carve gives sigma >= t_split. At t_split=0.92, nfe=50,
    # batch=256 that is 5; the old hardcoded 4 under-filled the band by ~20%.
    band_batch=os.environ.get('LAKON_BAND_BATCH', 'auto'),
    # band_rows: how many (trajectory, band-time) PAIRS the CFG-gap and emp_fm terms
    # score, drawn independently -- decoupled from band_batch (how many trajectories
    # are rolled out, one expert bank each). 'auto' = round(p_high * batch) = the
    # band's own point budget, so the band keeps its share of the objective; raise it
    # for a lower-variance estimate without paying for more banks.
    band_rows=os.environ.get('LAKON_BAND_ROWS', 'auto'),
    # Rollout classes drawn INDEPENDENTLY of the minibatch ('prior' = the dataset's
    # class counts, matching the on-path stream and the buffered DAGGER hook's
    # 'proportional'). Two consequences: the band is no longer confined to whatever
    # classes this batch happened to contain, and -- because the draw does not wait on
    # a future batch -- the next band's expert banks can be loaded a full interval
    # ahead (true double-buffering) instead of one FM step ahead.
    band_class_sampling=os.environ.get('LAKON_BAND_CLASSES', 'prior'),
    # Confine the band to this many DISTINCT classes (~band_batch/C trajectories each
    # instead of ~1). Unset -> unrestricted.
    band_classes_per_batch=(int(os.environ['LAKON_BAND_NCLS'])
                            if os.environ.get('LAKON_BAND_NCLS') else None),
    # Draw the NULL bank (the empirical v*_uncond) from the band's own classes rather
    # than from the full dataset. Only coherent alongside LAKON_BAND_NCLS: the target
    # gap v*_cond - v*_uncond then compares a class against the restricted mixture it
    # actually sits in, instead of against the 1000-class marginal.
    band_null_from_batch=bool(int(os.environ.get('LAKON_BAND_NULLBATCH', '0'))),
    band_sampler='FlowHeunODE',
    cfg_gap_max_states=int(os.environ.get('LAKON_BAND_STATES', 6)),
    cfg_gap_detach_uncond=False,
    mmd_interval=_bint,
    mmd_start_iter=500,       # same warmup as the DAGGER rounds
    mmd_eval_mode=True,
    mmd_accum_steps=1,        # no grad accumulation in this arm
))

resume_from = f'checkpoints/{name}/latest.pth'

# No replay buffer: the DaggerRolloutHook is dropped entirely (mmcv replaces lists),
# so nothing is ever stored or replayed and every band point is on-policy.
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

data = dict(train_dataloader=dict(samples_per_gpu=_spg))
