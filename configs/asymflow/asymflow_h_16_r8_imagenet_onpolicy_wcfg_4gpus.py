"""on-policy band arm, sweeping w_cfg on the FULL CFG term, with full-ish expert banks.

Expanding the band's gap term with a = v_cond - v*_cond, b = v_uncond - v*_uncond:

    ||a - b||^2  =  ||a||^2 + ||b||^2  -  2 <a, b>
                    \\__ per-branch FM __/   \\_ CFG alignment _/

w_cfg (LAKON_CFG_GAP_WEIGHT) scales the full CFG term at the on-policy band states:

    w_cfg * || (v_cond - v_uncond) - (v*_cond - v*_uncond) ||^2

i.e. it asks the model's CFG direction to match the empirical expert's, and is blind
to any error common to both branches. The per-branch magnitudes are pinned
separately by emp_fm (v_cond -> v*_cond, v_uncond -> v*_uncond) at its own fixed
weight, so the sweep moves the coupling ALONE:

    w_cfg = 0   -> emp_fm only: each branch regressed onto its own expert velocity
                   (on-policy DAGGER, no CFG coupling at all)
    w_cfg > 0   -> additionally pins the difference between the branches

Supervision is EXCLUSIVE by sigma: on-path FM owns sigma < band_t_split (the
inherited t_split carve, engaged automatically by onpath_truncate='auto' whenever
emp_fm is on), the band terms own sigma >= band_t_split. Supervising the band twice
-- once against the true residual noise - x_0 and once against v* -- was a bug in
the earlier align runs, which were discarded.

NB the two band terms are NOT on the same scale: emp_fm goes through flow_loss (the
logit-normal rescale, a heavy down-weight at these sigmas, plus its internal 0.5),
while the gap is a plain mean. So w_cfg = 1 is not "equal footing" -- read the
logged cfg_gap against emp_fm and loss_diffusion (~0.06) before trusting a range.

Logged every banded iteration: cfg_gap = the term actually optimised, plus the
diagnostics cfg_self = ||a||^2+||b||^2, cfg_cross = <a,b>, cfg_cos = their cosine
(a = v_cond - v*_cond, b = v_uncond - v*_uncond). cfg_cos is the one to watch -- if
the two residuals are uncorrelated it sits at ~0 and the coupling has nothing to
pin.

MMD is OFF (mmd_weight=0, inherited), so the rollout is pure inference under
no_grad and the band term is an ordinary per-point regression at detached states.
LR therefore stays at the comparison arms' 2.5e-4.

BANKS -- the cost driver. bank_size=None makes the conditional bank the ENTIRE class
(~1300 images); null_bank_size=2048, drawn per trajectory (not shared or cached), so
every banded iteration loads

    band_batch * (~1300 + 2048)  images  from GPFS at ~500 img/s/rank

  band_batch=16, mmd_interval=4  -> 53.6k imgs -> ~107 s/banded iter -> ~37 h / run
  band_batch=4,  mmd_interval=8  -> 13.4k imgs ->  ~27 s/banded iter ->  ~4.7 h / run

The defaults below take the second, because a SWEEP multiplies it by the number of
w values. The null bank dominates (2048 of the 3348), so it is the first knob to
move if this is still too slow. Measure the real rate with the smoke before
committing a sweep -- the 500 img/s figure is from the DAGGER bankfull runs.

Each swept value gets its OWN name/work_dir/checkpoints: a shared name would make
the runs resume from each other's latest.pth.
"""

import os

_base_ = ['./asymflow_h_16_r8_imagenet_onpolicy_4gpus.py']

# w_cfg: the coefficient on the FULL CFG term
#     w_cfg * || (v_cond - v_uncond) - (v*_cond - v*_uncond) ||^2
# (the cosine/cross decomposition this config used to sweep is gone -- cfg_self,
# cfg_cross and cfg_cos are still logged, but as diagnostics only.)
_w = os.environ.get('LAKON_CFG_GAP_WEIGHT', '1.0')

name = f'asymflow_h_16_r8_imagenet_onpolicy_wcfg{_w}_4gpus'
work_dir = f'work_dirs/{name}'

model = dict(
    expert=dict(
        bank_size=None,        # entire class per conditional bank
        null_bank_size=2048,   # per-trajectory null bank (x2 with flips), not shared
    ),
    diffusion=dict(
        # w_cfg on the full CFG term:
        #   w_cfg * || (v_cond - v_uncond) - (v*_cond - v*_uncond) ||^2
        cfg_gap_weight=float(_w),
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
        # cost knobs -- see the bank note above
        band_batch=int(os.environ.get('LAKON_BAND_BATCH', 4)),
        mmd_interval=int(os.environ.get('LAKON_BAND_INTERVAL', 1)),
        # score the whole band rather than a random 2 states: with band_batch this
        # small the extra states are the cheap way back to a usable gradient, and
        # they reuse the SAME banks (the expensive part) at no extra IO.
        cfg_gap_max_states=int(os.environ.get('LAKON_BAND_STATES', 6)),
    ),
)

resume_from = f'checkpoints/{name}/latest.pth'
