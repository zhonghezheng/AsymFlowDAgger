# Copyright (c) 2026 Hansheng Chen

import contextlib
import os
import time

from concurrent.futures import ThreadPoolExecutor

import torch

from ..builder import MODULES
from .gaussian_flow_mmd import GaussianFlowMMD


@MODULES.register_module()
class GaussianFlowOnPolicy(GaussianFlowMMD):
    """Fully-online on-policy training in the DAGGER band -- no replay buffer.

    The DAGGER arms visit the high-sigma band through a *replay buffer*: a rollout
    round every ``round_interval`` iterations fills ``model.dagger_buffer`` with
    expert-labelled states, and the next few hundred iterations resample it. The
    states therefore go stale between rounds (the policy that generated them is
    several hundred updates old) and every round pays for the expert's image banks.

    Here no STATE is stored. Each iteration rolls the CURRENT policy out from noise
    on the eval sampler/NFE grid, stops at the band edge, and trains at exactly the
    states that rollout visited -- so every point is on-policy for the weights being
    updated, and no rollout state is ever reused after the weights move.

    Two terms are available at those on-policy states, both optional:

    ``cfg_gap_weight`` (w_cfg) -- the CFG-gap matching loss

        || (v_cond - v_uncond) - (v*_cond - v*_uncond) ||^2

        where ``v`` is the model's velocity and ``v*`` the DATA-derived empirical
        velocity of the ground-truth model. The model's two branches come from one
        batched forward of ``2 * band_rows`` rows (true labels | null). The target
        is the empirical expert's gap at the same state: its class-restricted
        posterior mean gives ``v*_cond`` and its whole-dataset posterior mean gives
        ``v*_uncond``, both mapped through ``expert_target`` into the same
        clamp-weighted velocity space, and detached.

        So the band is not asked to make conditioning irrelevant -- it is asked to
        reproduce the CFG gap the data itself implies. The band is precisely the
        region inference leaves unguided (the eval ``guidance_interval`` upper edge
        is 0.88 <= band edge), which is where the model's gap is otherwise never
        corrected by guidance at sampling time.

        ``cfg_gap_target='zero'`` drops ``v*`` and drives the model's own gap to
        zero instead. That is a DIFFERENT objective (it makes the band
        class-agnostic rather than data-matched) and needs the expert for nothing,
        so it is available only as an explicit ablation.

    ``emp_fm_weight`` -- the empirical flow-matching loss at the same states,
        regressing each branch onto its OWN expert velocity (``v_cond -> v*_cond``,
        ``v_uncond -> v*_uncond``). This is DAGGER's ``loss_dagger`` target, taken at
        on-policy states instead of buffered ones, and it shares the gap term's model
        forward and expert evaluation -- so with both on, the second term is free
        apart from the loss arithmetic.

        It is complementary to the gap, not redundant with it. With
        a = v_cond - v*_cond and b = v_uncond - v*_uncond, the gap is ``||a - b||^2``
        while this is ``~ ||a||^2 + ||b||^2``: the gap constrains only the DIFFERENCE
        and is blind to error common to both branches, while this pins the absolute
        level of each. Together they fix both.

    ``mmd_weight`` -- the inherited per-step MMD between the rollout marginal and
        the noised on-path marginal (see :class:`GaussianFlowMMD`). Off by default
        (``mmd_weight=0``).

    Loss composition, as in d-flow (and DAGGER's proportional carve): the FM loss is
    ONE mean over the step's batch ``B``, and emp_fm's rows are part of it. A band
    iteration takes ``n_exp = round(p_high (1 - f) B)`` expert rows (``band_rows``)
    out of the micro-batch that runs the band; the on-path rows left stop at the band
    edge apart from ``round(p_high f mb)`` real-data rows inside it (``f`` =
    ``band_frac_on_path``, ``p_high = P(sigma >= t_split)`` under the uniform
    timestep sampler). Every row then weighs ``1/B``: the row counts carry reg_ft's
    sigma allocation, the flow loss's per-row logit-normal weight turns it into
    reg_ft's mass, and no per-term mass factor is needed. Rollout rows sit on the
    solver grid, so their weight is their grid cell's average, not the value at the
    state (see :meth:`_band_grid`). The CFG gap and MMD stay separate terms with
    their own weights. An iteration without a band (warmup, ``mmd_interval``) is
    plain full-range reg_ft.

    Gradient structure, which differs between the two on purpose:

      * the CFG-gap term is a per-point regression at *fixed* visited states, so the
        states are DETACHED before it runs -- the gradient goes into the two
        velocity predictions, never into the trajectory that produced the state.
        That is DAGGER's semantics (visit on-policy, regress pointwise) and it keeps
        the term's gradient independent of the rollout's.
      * MMD is a property of the rollout DISTRIBUTION, so it has no gradient at all
        unless the samples themselves depend on theta: it uses the graph-carrying
        states and backprops through every Heun step in the band.

    Consequently the rollout is built under ``no_grad`` when MMD is off (the common
    case: the CFG-gap term alone needs no trajectory graph), and with the graph only
    when ``mmd_weight != 0``. Turning MMD on is what makes the band expensive.

    Both the rollout and the point terms' forward run the net in EVAL mode -- the
    deterministic function the sampler uses -- so neither the visited states nor
    v_cond / v_uncond carry dropout noise; only the on-path FM rows train with dropout.

    The expert is still needed -- it supplies ``v*`` -- but only its banks, built
    fresh each iteration the band runs and never cached across weight updates. What
    is gone is the replay BUFFER of rollout states: the ``DaggerRolloutHook`` must
    not be used and a ``buffer_batch`` is rejected, since states generated by an
    older policy are the thing this class exists to remove. Bank construction is the
    arm's dominant cost -- see :meth:`_build_gap_banks`.

    Args:
        cfg_gap_weight (float): coefficient on the mean-over-band gap-matching loss.
            0 disables the term. The mean is over elements and band rows, with no
            logit-normal reweighting unless ``cfg_gap_reweight`` -- a plain mean
            keeps the knob's scale readable against the logged value, but note the
            weight is far from constant across the band (0.82 at sigma 0.92, 0.015
            at 0.98), so the two choices emphasise different ends of it.
        emp_fm_weight (float): per-row multiplier on the empirical FM rows inside the
            shared FM mean (see "Loss composition"). 1 (default) = parity: an expert
            row weighs exactly what the on-path row it replaced would have. 0
            disables the term (and the carve). Unlike the gap, its rows take
            ``self.flow_loss``'s per-row value (logit-normal rescale and the 0.5
            factor included), so the logged per-row ``emp_fm`` is directly comparable
            to the DAGGER arms' ``loss_dagger`` -- and is much smaller than a raw MSE
            at these sigmas, where that reweighting down-weights hard.
        cfg_gap_target (str): ``'expert'`` (default) matches the data-derived gap
            ``v*_cond - v*_uncond``; ``'zero'`` is the expert-free ablation that
            drives the model's own gap to zero.
        cfg_gap_detach_uncond (bool): stop-gradient on the MODEL's unconditional
            branch, so the term moves only the conditional field and leaves the
            unconditional one where it is. Default False (both differentiated, so
            the pair meets the target from both sides).
        null_label (int | None): label index for the unconditional branch. ``None``
            -> ``denoising.num_classes`` (the CFG embedding slot, 1000 here), which
            is what the wrapper's ``negative_labels`` carries.
        band_t_split (float | None): band lower edge. ``None`` falls back to
            ``mmd_t_split``, then to ``t_split``.
        band_nfe (int | None), band_sampler (str | None): rollout grid /
            scheduler. ``None`` keeps the inherited ``mmd_nfe`` / ``mmd_sampler``.
            One band, one rollout: every term reads the same states, so these
            override the MMD names rather than sitting beside them.
        band_rows (int | str): how many rows the point terms score per band, ONE
            per trajectory: the band rolls out exactly this many trajectories
            (plus the t=1-only ones of ``band_score_start``) and scores each at one
            band state drawn uniformly, so no two rows share a trajectory and no
            rollout goes unscored. Each row costs one bank. ``'auto'`` = the expert's
            row share of the step's batch, ``round(p_high * (1 - band_frac_on_path)
            * batch)`` (``round(p_high * batch)`` with emp_fm off, where the rows
            serve the gap / MMD alone). An explicit count changes the band's share
            of the FM mean with it.
        band_score_start (bool): also score the point terms (CFG gap and
            emp_fm) at the t=1 START state of the rollout (pure noise -- inference's
            first eval), not only at the landing states. The gap has a real target
            there: the expert's softmax is flat at t=1, so v*_c - v*_u = x0_u - x0_c,
            the class-mean offset. Start rows are added at the same per-state share as
            each landing state (round(band_rows / n_states)), each on a trajectory of
            its OWN -- fresh noise, never rolled out, since the start is its only
            scored state -- so each costs one bank. emp_fm leaves them out: they
            are not carved rows, and its logit-normal weight at t=1 is ~1e-52
            anyway. MMD leaves the start out too -- both of its sides are exactly
            N(0, I) there, so the statistic is identically 0. Default False.
        band_states (str): where the band's scored states come from. ``'rollout'``
            (default) -- the current policy's Heun rollout, as described above.
            ``'onpath'`` -- NO rollout: each row is a real image noised to a
            sigma drawn from the training timestep sampler restricted to
            sigma >= the band edge, ``x_t = (1 - s) x0 + s eps``. That is reg_ft's
            band exactly, with the true velocity swapped for v* (the regft_emp arm).
            The image is a uniform entry of the row's own class bank (a uniform
            image of its class, either orientation -- the dataloader's distribution
            under 'prior' class sampling), and BOTH orientations of it are left out
            of that row's conditional posterior. Without that the posterior at
            sigma >= 0.88 collapses onto x0 itself and v* reduces to eps - x0, i.e.
            to reg_ft. Needs the u8 bank storage; MMD and
            band_score_start have no meaning without a rollout and are refused.
        band_onpath_keep (bool): ``'onpath'`` only -- KEEP each row's image in its
            posteriors instead of leaving it out: no exclusion in the class bank, and
            the image is written into the row's null bank too. v* is then the
            empirical velocity of a distribution that contains x0 (at T=1 and
            sigma <~ 0.94 it is eps - x0 to within float error). Default False.
        cfg_gap_reweight (bool): weight the CFG gap per row by the flow loss's own
            sigma weight (0.5 * its rescale -- the pretraining logit-normal mass),
            exactly as emp_fm and the on-path FM are (cell-averaged on grid rows),
            instead of a plain mean. Under the logit-normal(0.8, 0.8) weight the
            rows at 0.98 / t=1 then carry ~0.003 / ~0. The unweighted value is still
            logged as cfg_gap_raw. Default False.
    """

    def __init__(self,
                 *args,
                 cfg_gap_weight=1.0,
                 cfg_gap_target='expert',
                 emp_fm_weight=1.0,
                 cfg_gap_detach_uncond=False,
                 null_label=None,
                 band_prob_class=0.9,
                 emp_fm_complement_mode=None,
                 band_null_from_batch=False,
                 band_frac_on_path=0.5,
                 allow_no_band_term=False,
                 onpath_truncate='auto',
                 band_t_split=None,
                 band_nfe=None,
                 band_rows='auto',
                 band_class_sampling='uniform',
                 band_classes_per_batch=None,
                 band_sampler=None,
                 band_score_start=False,
                 band_states='rollout',
                 band_onpath_keep=False,
                 cfg_gap_reweight=False,
                 **kwargs):
        kwargs.setdefault('mmd_weight', 0.0)   # MMD is opt-IN here, unlike GaussianFlowMMD
        super().__init__(*args, **kwargs)
        self.cfg_gap_weight = float(cfg_gap_weight)
        assert cfg_gap_target in ('expert', 'zero')
        self.cfg_gap_target = cfg_gap_target
        assert emp_fm_weight != 'mass', (
            "emp_fm_weight='mass' is gone: the band's sigma mass now comes from its ROW "
            'count in the shared FM mean (see "Loss composition"), so 1.0 is parity.')
        self.emp_fm_weight = float(emp_fm_weight)
        self.cfg_gap_detach_uncond = cfg_gap_detach_uncond
        self.null_label = int(null_label) if null_label is not None \
            else int(self.denoising.num_classes)
        # one band shared by every on-policy term -- these overwrite the inherited
        # mmd_* fields so the inherited _band_rollout / _ckpt_pred need no changes.
        if band_t_split is not None:
            self.mmd_t_split = band_t_split
        if band_nfe is not None:
            self.mmd_nfe = int(band_nfe)
        # band_rows: how many rows the point terms score, one per trajectory (see
        # _draw_band_plan) -- so it is also the rollout width and the bank count.
        self.band_rows = band_rows
        if band_sampler is not None:
            self.mmd_sampler = band_sampler
        self.band_score_start = bool(band_score_start)
        assert band_states in ('rollout', 'onpath'), band_states
        self.band_states = band_states
        assert not band_onpath_keep or band_states == 'onpath', \
            "band_onpath_keep only applies to band_states='onpath'."
        self.band_onpath_keep = bool(band_onpath_keep)
        if band_states == 'onpath':
            assert self.mmd_weight == 0 and not self.band_score_start, (
                "band_states='onpath' has no rollout: MMD and band_score_start (the "
                'rollout start state) do not apply.')
        self.cfg_gap_reweight = bool(cfg_gap_reweight)
        # On-path FM must STOP at the band edge whenever an on-policy term already
        # supervises the band: emp_fm regresses both branches onto the expert's v*
        # there, so leaving the FM loss full-range supervises the same sigmas twice --
        # once against the true residual noise - x_0 and once against v* -- and the
        # arm's sigma mass stops matching reg_ft. 'auto' truncates exactly when
        # emp_fm is on (True/False force it), mirroring the DAGGER arms' t_split
        # carve; the inherited _sample_t_below_split reads self.t_split, so the band
        # edge is copied there.
        # CFG dropout for the band's empirical-FM term: each on-policy point is
        # regressed onto ONE branch's expert velocity -- the conditional one with
        # probability band_prob_class, the unconditional one otherwise. This mirrors
        # the DAGGER arms' convention (DaggerRolloutHook's label_time_dropout, at
        # train_cfg.prob_class), so emp_fm here is the same quantity as their
        # loss_dagger rather than a both-branches average. The CFG-gap term is
        # unaffected: it needs v_cond AND v_uncond at every point to form a
        # difference, and a dropped row would compare the unconditional branch
        # against itself and contribute an exact zero.
        assert 0.0 <= band_prob_class <= 1.0
        self.band_prob_class = float(band_prob_class)
        # band_frac_on_path: fraction of the BAND's rows that are REAL data (on-path FM
        # with the true velocity) rather than expert rows at on-policy states --
        # DAGGER's frac_on_path, applied the same way, as a row split of the carve
        # (see _carve_counts). Without it the band has no ground-truth anchor at all
        # and v* (a smoothed bank average) owns the region outright, which is a
        # plausible driver of the mode narrowing those runs showed.
        # emp_fm keeps BOTH halves of its target -- subspace AND complement --
        # always. emp_fm_complement_mode selects how the COMPLEMENT is computed
        # (None -> inherit diffusion.complement_mode):
        #
        #   'full'    : the complement of the plain velocity toward x0_hat,
        #               P_perp((x_t - x0_hat)/sigma_c). The form the network's
        #               complement head natively produces (its ideal output there is
        #               -x0_hat_comp), which is why every arm so far has used it.
        #   'project' : the low-rank prior, -clamp_coef * x0_hat_comp. The subspace
        #               half is IDENTICAL either way (verified to 1.7e-5); only the
        #               complement differs, and the head can only reach this one by
        #               cancelling its own hard-wired x_t_comp term.
        #
        # 'project' on the DAGGER regression target diverged badly once (loss_dagger
        # stuck at 0.73, FID 4.32 -> 294.6).
        #
        # The CFG gap has no such option: it is always 'full', the assembled velocity
        # space self.pred() returns (expert_target('full') is the assembled image of
        # the derived empirical asym target). There x_t cancels between the branches,
        # so the target gap is exactly (x0_hat_uncond - x0_hat_cond) / sigma_c.
        self.emp_fm_complement_mode = emp_fm_complement_mode
        # Draw the null bank from the batch's own classes rather than the whole
        # dataset. See EmpiricalExpert.build_banks for why this is off by default;
        # it exists so the choice can be measured against the standard arms.
        self.band_null_from_batch = bool(band_null_from_batch)
        self._bank_pool = None      # single orchestrator thread issuing the bank IO
        self._bank_pending = None   # in-flight bank draw for THIS iteration
        self._bank_ready = None     # (plan, future) prefetched for the NEXT band
        self._band_t = {}           # this band's timings / bank costs, see _log_bank_cost
        self._grid_key = None       # cache for _band_grid
        self._grid = None
        # How the rollout's classes are drawn. 'batch' reuses the minibatch's labels
        # (what this arm did before); 'uniform'/'prior' draw them INDEPENDENTLY of the
        # batch, which is what the buffered DAGGER hook did and what makes real
        # double-buffering possible: a draw that does not depend on a future batch can
        # be issued a whole band interval ahead instead of one iteration ahead.
        assert band_class_sampling in ('batch', 'uniform', 'prior')
        self.band_class_sampling = band_class_sampling
        # Confine the band's trajectories to this many DISTINCT classes (the CFG-side
        # analogue of mmd_classes_per_batch). Unset -> whatever band_class_sampling
        # draws, which over 1000 classes gives ~1 trajectory per class. Set to e.g. 8
        # so each class carries several trajectories.
        #
        # Pair it with band_null_from_batch=True: the empirical v*_uncond is then an
        # average over images from the SAME restricted class set, so the target gap
        # v*_cond - v*_uncond is the gap of the restricted mixture rather than of the
        # full 1000-class marginal. Leaving the null unrestricted while restricting
        # the conditional side makes the two halves of the target gap refer to
        # different universes.
        self.band_classes_per_batch = band_classes_per_batch
        assert 0.0 <= band_frac_on_path <= 1.0
        self.band_frac_on_path = float(band_frac_on_path)
        # permits the w=0 control: every band weight 0, so the arm reduces to the
        # inherited DAGGER objective while still running through this class's code
        # path (which is the point of the control).
        self.allow_no_band_term = bool(allow_no_band_term)
        assert onpath_truncate in ('auto', True, False)
        self.onpath_truncate = onpath_truncate
        if self._truncate_onpath():
            assert self.mmd_t_split is not None, (
                'onpath_truncate needs a band edge: set band_t_split (or '
                'mmd_t_split) so the on-path FM knows where to stop.')
            self.t_split = self.mmd_t_split
        # onpath: the band rows ARE the sigma >= edge share of reg_ft's batch, so the
        # dataloader rows must stop at the edge (and the rows' sigma draw reads
        # self.t_split as that edge)
        assert self.band_states != 'onpath' or self._truncate_onpath(), (
            "band_states='onpath' needs the on-path FM truncated at the band edge "
            "(onpath_truncate True, or 'auto' with emp_fm on).")
        # the carve takes its expert rows out of the micro-batch the FM loss runs on,
        # so emp_fm needs the on-path rows to stop at the band edge to make room
        assert self.emp_fm_weight == 0 or self._truncate_onpath(), (
            'emp_fm needs onpath_truncate on: its rows replace the band share of the '
            'on-path batch, which must therefore stop at the band edge.')
        # NB the expert's presence is NOT checked here: the wrapper attaches the
        # handle (_dagger_expert) after the diffusion is constructed, so the check
        # lives in _build_gap_banks, at first use.
        assert (self.cfg_gap_weight != 0 or self.emp_fm_weight != 0
                or self.mmd_weight != 0 or self.allow_no_band_term), (
            'GaussianFlowOnPolicy with cfg_gap_weight, emp_fm_weight and mmd_weight '
            'all 0 has no on-policy term at all -- that is plain reg_ft, so use '
            'GaussianFlow (or the regft config) instead of paying for the band '
            'rollout.')

    def _truncate_onpath(self):
        """Whether the on-path FM loss is restricted to sigma < the band edge."""
        if self.onpath_truncate == 'auto':
            return self.emp_fm_weight != 0
        return bool(self.onpath_truncate)

    def _band_due(self, running_status):
        """Per-iteration gate + micro-batch latch for the band, mirroring
        :meth:`GaussianFlowMMD._mmd_due` but keyed on EITHER on-policy term
        term being enabled. The inherited version short-circuits on ``mmd_weight == 0``,
        which is this class's default (MMD is opt-in), so it would switch the whole
        band off whenever only the CFG-gap term is running."""
        if (self.cfg_gap_weight == 0 and self.emp_fm_weight == 0
                and self.mmd_weight == 0):
            return False
        if running_status is None:
            return True
        it = running_status.get('iteration', 0)
        return it >= self.mmd_start_iter and it % self.mmd_interval == 0

    def _band_claim(self, running_status):
        """True at most ONCE per iteration -- the band runs on one micro-batch. Split
        from _band_due so that merely ASKING whether the band is due cannot burn the
        latch (see GaussianFlowMMD._mmd_claim for the bug that split caused)."""
        if not self._band_due(running_status):
            return False
        if running_status is None:
            return True
        it = running_status.get('iteration', 0)
        if self._mmd_done_iter == it:
            return False
        self._mmd_done_iter = it
        return True

    # ---- the on-policy band terms -------------------------------------------

    def _expert_velocities(self, x_s, sigma, labels, banks, exclude=None):
        """The DATA-derived velocities ``(v*_cond, v*_uncond)`` at one visited state.

        Both come from the empirical expert's posterior-mean data estimate at ``x_s``,
        differing only in which bank each row averages over:

          * ``v*_cond``   -- row ``i`` over its CONDITIONAL bank (``bank_size`` draws
            from row ``i``'s true class), i.e. the class-restricted posterior.
          * ``v*_uncond`` -- the same rows over their NULL bank (``null_bank_size``
            draws from the whole dataset), i.e. the unconditional posterior.

        ``x0_hat`` selects per row by label, so passing the true labels takes the
        conditional banks and passing all-null takes the null banks -- two calls over
        the SAME ``build_banks`` output, no second load. ``expert_target`` then maps
        each posterior mean to a velocity in the clamp-weighted space the model's own
        ``u_t_pred`` lives in, so model and expert velocities are comparable.

        These are fixed data-derived targets: they carry no gradient. ``exclude``
        (per row, entries of its CONDITIONAL bank to leave out) applies to v*_cond
        only: the indices mean nothing in the null bank.
        """
        expert = self._dagger_expert
        cond_banks, null_banks = banks
        # sigma arrives as the per-row vector the caller already built: with one band
        # state drawn per trajectory the rows sit at DIFFERENT sigmas, and x0_hat
        # scores each row at its own noise level.
        t0 = self._synced_clock(x_s.device)
        with torch.no_grad():
            x0_cond = expert.x0_hat(
                x_s, sigma, self.feat_fn, labels,
                cond_banks, null_banks, self.null_label, exclude=exclude)
            t1 = self._synced_clock(x_s.device)
            x0_uncond = expert.x0_hat(
                x_s, sigma, self.feat_fn, torch.full_like(labels, self.null_label),
                cond_banks, null_banks, self.null_label)
        t2 = self._synced_clock(x_s.device)
        bt = self._band_t
        bt['band_t_x0hat_cond'] = bt.get('band_t_x0hat_cond', 0.0) + (t1 - t0)
        bt['band_t_x0hat_null'] = bt.get('band_t_x0hat_null', 0.0) + (t2 - t1)
        # return the POSTERIOR MEANS, not velocities: the two band terms may map them
        # through different complement modes, and expert_target is pure arithmetic
        # (the bank averaging above is the expensive part, done once).
        return x0_cond.detach(), x0_uncond.detach()

    @contextlib.contextmanager
    def _dropout_off(self):
        """The band's point-term forward in EVAL mode: no dropout, the function the
        sampler runs (and the rollout's _ckpt_pred already uses). In train mode each
        half of the [cond; null] forward draws its own dropout mask, which adds
        Var_m(v_c) + Var_m(v_u) to the gap and to emp_fm as a floor no update can
        remove (tools/cfg_gap_floor.py measures it). Eval also swaps AsymJiT's
        assembly clamp to its inference sigma_min, which acts only far below the band.

        Backward safety: the net checkpoints per block, and a checkpoint recomputes
        DURING loss.backward(), after train mode is back. Under torch.compile the
        recompute is part of the compiled backward, traced here in eval mode, so it is
        exact. Eager has no such graph -- it would re-run the blocks with dropout on
        -- so there the per-block checkpointing is off for this forward instead (only
        the band's 2 * band_rows rows)."""
        net = self.denoising
        if not net.training:
            yield
            return
        eager_ckpt = (getattr(net, '_compiled_forward', None) is None
                      and getattr(net, 'gradient_checkpointing', False))
        net.eval()
        if eager_ckpt:
            net.gradient_checkpointing = False
        try:
            yield
        finally:
            net.train()
            if eager_ckpt:
                net.gradient_checkpointing = True

    def _loss_row_weights(self, t):
        """The flow loss's per-row weight at timesteps ``t``: what it multiplies a row's
        flat-mean squared error by (its internal 0.5, then rescale_fn). A plain per-row
        MSE times this, averaged over rows, is exactly flow_loss's value."""
        return self.flow_loss.rescale_fn(torch.full_like(t, 0.5), t)

    def _band_point_losses(self, x_s, sigma, labels, banks=None, exclude=None,
                           row_w=None):
        """Both on-policy point terms at ONE visited state, sharing one model forward
        and one expert evaluation. Returns ``(gap, fm_rows)``: ``fm_rows`` holds each
        row's flow-loss value (``None`` when the empirical FM term is off), for the
        caller to SUM into the shared FM mean. ``row_w`` multiplies the flow loss's
        per-row sigma weight (the grid rows' cell correction, see :meth:`_band_grid`)
        wherever that weight is used.

        The model's two branches come from a single batched forward over ``[x_s; x_s]``
        with ``[labels; null]`` in eval mode (no dropout, see :meth:`_dropout_off`), so
        the pair is evaluated by the deterministic net inference runs and costs one
        forward of ``2m`` rows rather than two of ``m``. Velocities are
        taken in GaussianFlow's clamp-weighted space (``output * clamp_coef``, the
        ``u_t_pred`` the flow loss regresses), which is also what ``expert_target``
        returns.

        gap -- ``|| (v_cond - v_uncond) - (v*_cond - v*_uncond) ||^2``. Depends only on
            the DIFFERENCE of the two branches, so it is blind to any error common to
            both: a model whose velocities are uniformly offset from the data has zero
            gap loss.
        fm  -- the empirical flow-matching loss. Each point is regressed onto ONE
            branch's expert velocity under CFG dropout: ``v_cond -> v*_cond`` with
            probability ``band_prob_class``, else ``v_uncond -> v*_uncond``. That is
            exactly the DAGGER arms' ``loss_dagger`` convention (one label, one
            target per visited state), evaluated at on-policy states.

        The two are complementary rather than redundant: writing a = v_cond - v*_cond
        and b = v_uncond - v*_uncond, gap = ||a - b||^2 while fm ~ ||a||^2 + ||b||^2.
        The gap pins the CFG-relevant difference and tolerates common-mode error; fm
        pins the absolute level. Together they fix both.

        NB the two use DIFFERENT weightings on purpose: gap is a plain mean (its
        scale stays readable against cfg_gap_weight), while fm takes ``self.flow_loss``'s
        per-row value -- logit-normal rescale and the 0.5 factor included -- so a band
        row weighs exactly what an on-path row at the same sigma does, and the logged
        value is directly comparable to the DAGGER arms' loss_dagger. That reweighting
        is a strong down-weight at these sigmas, so expect fm to log much smaller than
        a raw MSE would.
        """
        m = x_s.size(0)
        # sigma may be a scalar (one state for the whole batch) or a per-ROW tensor
        # (one uniformly-drawn band state per trajectory -- the default sampling).
        sig_vec = sigma.to(x_s).reshape(m) if torch.is_tensor(sigma) \
            else x_s.new_full((m, ), float(sigma))
        t = sig_vec * self.num_timesteps
        # the flow loss's per-row sigma weight, cell-corrected on grid rows
        row_lw = self._loss_row_weights(t)
        if row_w is not None:
            row_lw = row_lw * row_w.to(row_lw)
        both = torch.cat([x_s, x_s], dim=0)
        cond_null = torch.cat(
            [labels, torch.full_like(labels, self.null_label)], dim=0)
        with self._dropout_off():
            out = self.pred(both, torch.cat([t, t], dim=0), class_labels=cond_null)
        _, _, clamp_coef = self.get_clamp_coef(t=t, x_t=x_s)
        v_cond, v_uncond = out.chunk(2, dim=0)
        v_cond = v_cond * clamp_coef
        v_uncond = v_uncond * clamp_coef

        vs_cond = vs_uncond = vs_fm_c = vs_fm_u = None
        if banks is not None:
            x0_c, x0_u = self._expert_velocities(x_s, sig_vec, labels, banks, exclude)
            # the CFG gap is always 'full'; emp_fm takes its own complement treatment
            vs_cond = self.expert_target(x_s, sig_vec, x0_c, mode='full')
            vs_uncond = self.expert_target(x_s, sig_vec, x0_u, mode='full')
            vs_fm_c = self.expert_target(x_s, sig_vec, x0_c, mode=self.emp_fm_complement_mode)
            vs_fm_u = self.expert_target(x_s, sig_vec, x0_u, mode=self.emp_fm_complement_mode)

        gap = fm_rows = None
        if self.cfg_gap_weight != 0:
            # the detach knob applies to the GAP only: the fm term must keep the
            # gradient on both branches, or it would never train the unconditional
            # field toward its own expert target.
            a = v_cond
            b = v_uncond.detach() if self.cfg_gap_detach_uncond else v_uncond
            if vs_cond is not None:              # 'zero' target leaves v* out (a=v_cond)
                a = a - vs_cond
                b = b - vs_uncond
            # The CFG term, taken directly:
            #   || (v_cond - v_uncond) - (v*_cond - v*_uncond) ||^2  ==  || a - b ||^2
            # scaled by cfg_gap_weight (w_cfg) where it is added to the loss.
            # cfg_gap_reweight: the flow loss's own per-row sigma weight (0.5 * its
            # rescale, i.e. the pretraining logit-normal mass), so the gap is weighted
            # along sigma exactly as emp_fm and the on-path FM are. None -> plain mean.
            rw = row_lw if self.cfg_gap_reweight else None
            wmean = (lambda rows: (rw * rows).mean()) if rw is not None else \
                (lambda rows: rows.mean())
            if rw is not None:
                gap = wmean((a - b).pow(2).flatten(1).mean(1))
            else:
                gap = (a - b).pow(2).mean()
            # Diagnostic split of the gap across AsymJiT's rank-8 basis. The loss
            # is over all 196608 dims, of which the subspace is 1.04%, so it is
            # numerically complement-dominated -- but that is only a problem if
            # the complement's share is noise rather than signal. Synthetic
            # modelling says signal and bank-sampling noise follow the SAME
            # spectrum (the error of a bank average IS the data covariance / n),
            # leaving SNR roughly flat across the split; these two scalars test
            # that on real data. gap == cfg_gap_sub + cfg_gap_comp exactly, the
            # projection being orthogonal.
            with torch.no_grad():
                g = a - b
                g_sub = self.project_fn(g)
                # diagnostics carry the same row weighting as the loss, so
                # cfg_gap == cfg_gap_sub + cfg_gap_comp still holds
                gap_sub = wmean(g_sub.pow(2).flatten(1).mean(1))
                gap_comp = wmean((g - g_sub).pow(2).flatten(1).mean(1))
                # per-row gap (weighted likewise, so gap == gap_rows.mean()), for
                # logging the t=1 rows apart from the landing ones
                gap_rows = g.pow(2).flatten(1).mean(1)
                gap_raw = gap_rows.mean()                # unweighted, comparable to old runs
                if rw is not None:
                    gap_rows = rw * gap_rows
        fm_cond_frac = emp_subfrac = None
        if self.emp_fm_weight != 0:
            # per-row CFG dropout: keep -> regress the CONDITIONAL branch onto
            # v*_cond, else the UNCONDITIONAL branch onto v*_uncond. One branch per
            # point, so this is DAGGER's loss_dagger taken on-policy, not an average
            # over both branches.
            keep = torch.rand(m, device=v_cond.device) < self.band_prob_class
            sel = keep.reshape(m, *([1] * (v_cond.dim() - 1)))
            r = torch.where(sel, v_cond, v_uncond) - torch.where(sel, vs_fm_c, vs_fm_u)
            # flow_loss's per-row value: flat-mean squared error times its row weight
            fm_rows = r.float().pow(2).flatten(1).mean(1) * row_lw
            with torch.no_grad():   # where emp_fm's residual sits, as for the gap
                r = r.detach()
                r_sub = self.project_fn(r)
                emp_subfrac = r_sub.pow(2).mean() / r.pow(2).mean().clamp_min(1e-12)
            fm_cond_frac = keep.float().mean().detach()
            emp_subfrac = emp_subfrac.detach()
        return gap, fm_rows, (fm_cond_frac, emp_subfrac), \
            (gap_sub, gap_comp, gap_rows, gap_raw) if gap is not None else None

    def _draw_band_labels(self, n, device, batch_labels=None):
        """The rollout's classes. Independent of the minibatch unless
        ``band_class_sampling='batch'``; 'prior' follows the dataset's class counts
        (the DAGGER hook's 'proportional'), 'uniform' is flat over the classes."""
        if self.band_class_sampling == 'batch':
            assert batch_labels is not None
            return batch_labels[:n]
        expert = self._dagger_expert
        if self.band_class_sampling == 'prior' and expert is not None \
                and hasattr(expert, 'sample_labels'):
            lab = expert.sample_labels(n, device)
        else:
            lab = torch.randint(0, int(self.denoising.num_classes), (n, ), device=device)
        return self._restrict_classes(lab, device)

    def _restrict_classes(self, lab, device):
        """Collapse a label draw onto ``band_classes_per_batch`` DISTINCT classes.

        The distinct set is taken from the draw itself rather than resampled flat, so
        which classes appear still follows ``band_class_sampling`` ('prior' keeps the
        dataset's class frequencies); only how MANY distinct ones survive changes.
        The surviving classes are then tiled over the rows, giving ~n/C trajectories
        each instead of ~1.
        """
        if self.band_classes_per_batch is None:
            return lab
        n = lab.numel()
        c = max(1, min(int(self.band_classes_per_batch), n))
        uniq = torch.unique(lab)
        if uniq.numel() > c:
            uniq = uniq[torch.randperm(uniq.numel(), device=device)[:c]]
        return uniq[torch.arange(n, device=device) % uniq.numel()]

    def _band_row_count(self, n_batch):
        """How many (trajectory, band-time) pairs the point terms score per step.

        ``n_batch`` is the FULL micro-batch, not the trajectory slice; the step's
        batch is ``n_batch * mmd_accum_steps`` (the band runs on one micro-batch per
        step), so the count does not move with gradient accumulation.

        ``'auto'`` is the expert's row share of that batch under the carve,
        ``round(p_high * (1 - band_frac_on_path) * batch)`` -- the rows that make
        every FM row of the step weigh the same (see :meth:`_carve_counts`). With
        emp_fm off the rows displace nothing and serve the CFG gap / MMD alone, so
        they take the whole band's ``round(p_high * batch)``.
        """
        if self.band_rows == 'auto':
            share = self.high_sigma_fraction()
            if self.emp_fm_weight != 0:
                share *= 1.0 - self.band_frac_on_path
            return max(1, int(round(share * n_batch * self.mmd_accum_steps)))
        return max(1, int(self.band_rows))

    def _carve_counts(self, n_batch, n_expert):
        """Row split of one micro-batch under the carve: ``(n_on, n_above)``.

        The FM loss is ONE mean over the micro-batch, as in d-flow and DAGGER's
        proportional carve: the band's ``n_expert`` rows replace that many on-path
        rows, and of the ``n_on`` on-path rows left, ``n_above`` are the real-data
        share of the band (``band_frac_on_path``, sigma >= t_split) and the rest sit
        below the edge. Every row then weighs ``1 / batch`` after accumulation, and
        the row counts alone carry reg_ft's sigma allocation -- the per-row
        logit-normal weight inside flow_loss turns them into its mass. ``n_above``
        is taken in every micro-batch of a band iteration, the expert rows only in
        the one that runs the band.
        """
        n_on = n_batch - n_expert
        n_above = int(round(self.high_sigma_fraction() * self.band_frac_on_path * n_batch))
        assert 0 < n_on and n_above <= n_on, (
            f'the carve needs {n_expert} expert + {n_above} in-band on-path rows out of '
            f'a {n_batch}-row micro-batch; lower band_rows or the accumulation.')
        return n_on, n_above

    def _band_grid(self, x_ref, t_split):
        """The band's solver grid, read off WITHOUT rolling out (same loop as
        :meth:`_band_rollout`): ``(n_states, cell_w)``. ``n_states`` lets the scored
        (trajectory, band time) pairs be drawn before the rollout (checked against it
        at use).

        ``cell_w[k]`` corrects the flow loss's sigma weight at band state ``k`` from
        its value AT the state to the mass of the state's CELL. reg_ft draws sigma
        continuously; a rollout row sits on a grid state and stands for the cell
        ``[sigma_k, sigma_{k-1})`` up to the state above it (1 for the first, capped
        at ``mmd_t_hi``; the lowest cell runs down to ``t_split``), so the cells tile
        the band. With states drawn uniformly, the row weight that reproduces reg_ft
        is the cell's integrated weight over the MEAN cell width. Read at the state
        instead, the steep logit-normal (0.82 at sigma 0.92, 0.015 at 0.98) is taken
        at the cells' lower edges and overstates the band by ~1.44x. Exactly 1 for
        a sigma-independent weight.
        """
        seq_len = x_ref.shape[2:].numel()
        key = (seq_len, float(t_split), self.mmd_nfe, self.mmd_t_hi, self.mmd_sampler)
        if self._grid_key != key:
            grid = self._build_mmd_sampler(seq_len, x_ref.device).sigmas.double().cpu()
            n_pre = 0
            if self.mmd_t_hi is not None:
                while 2 * (n_pre + 1) < len(grid) \
                        and float(grid[2 * (n_pre + 1)]) > self.mmd_t_hi:
                    n_pre += 1
            n_band = n_pre
            while 2 * (n_band + 1) < len(grid) \
                    and float(grid[2 * (n_band + 1)]) >= t_split:
                n_band += 1
            sig, mass, width = [], [], []
            for n in range(n_pre, n_band):
                s = float(grid[2 * (n + 1)])
                hi = float(grid[2 * n])
                if self.mmd_t_hi is not None:
                    hi = min(hi, self.mmd_t_hi)
                lo = float(t_split) if n == n_band - 1 else s
                # midpoint rule over the cell; the weight is smooth, so 1024 is plenty
                edges = torch.linspace(lo, hi, 1025, dtype=torch.float64)
                mids = (edges[1:] + edges[:-1]) / 2
                sig.append(s)
                mass.append(float(self._sigma_weight(mids).mean()) * (hi - lo))
                width.append(hi - lo)
            sig = torch.tensor(sig, dtype=torch.float64)
            mean_width = sum(width) / max(len(width), 1)
            cell_w = torch.tensor(mass, dtype=torch.float64) / mean_width \
                / self._sigma_weight(sig).clamp_min(1e-300)
            self._grid_key, self._grid = key, (n_band - n_pre, cell_w)
        return self._grid

    def _sigma_weight(self, sigma):
        """The flow loss's per-row sigma weight (its rescale_fn) at ``sigma``."""
        sigma = torch.as_tensor(sigma, dtype=torch.float64)
        return self.flow_loss.rescale_fn(torch.ones_like(sigma), sigma * self.num_timesteps)

    def _n_band_states(self, x_ref, t_split):
        """How many states :meth:`_band_rollout` returns (see :meth:`_band_grid`)."""
        return self._band_grid(x_ref, t_split)[0]

    def _needs_banks(self):
        """Whether any enabled term reads v*: the fm term always does, the gap
        unless running the expert-free 'zero' ablation."""
        return self.emp_fm_weight != 0 or (
            self.cfg_gap_weight != 0 and self.cfg_gap_target == 'expert')

    def _draw_band_plan(self, x_0, t_split, batch_labels=None):
        """Everything random about one band BEFORE it runs: the rollout's classes and
        the scored (trajectory, band time) pairs, so the bank IO can be issued ahead.

        ONE scored row per trajectory: ``n_rows`` trajectories, row ``j`` on trajectory
        ``j`` at a band state drawn uniformly (the band's t-marginal is uniform over
        the grid), and the t=1 rows of ``band_score_start`` on ``n_start`` further
        trajectories that are never rolled out (``n_roll`` marks the split). One bank
        per trajectory, ``traj`` = all of them.
        """
        n_states = self._n_band_states(x_0, t_split)
        n_rows = min(self._band_row_count(x_0.size(0)), x_0.size(0))
        n_start = max(1, int(round(n_rows / max(n_states, 1)))) \
            if self.band_score_start else 0
        n_start = min(n_start, x_0.size(0) - n_rows)
        labels = self._draw_band_labels(n_rows + n_start, x_0.device, batch_labels)
        plan = dict(labels=labels, j_pick=list(range(n_rows)),
                    k_pick=torch.randint(n_states, (n_rows, )).tolist(),
                    n_states=n_states, traj=list(range(n_rows + n_start)),
                    n_roll=n_rows)
        if n_start:
            plan['j_start'] = list(range(n_rows, n_rows + n_start))
        return plan

    def _draw_plan_paths(self, plan):
        """The bank path draw for ``plan`` (both the prefetch and the synchronous
        fallback go through here). Under ``band_states='onpath'`` it also fixes each
        row's image -- see :meth:`_place_onpath_images`."""
        labels = plan['labels']
        # the null restriction still follows the rollout's FULL class set, not just
        # the scored trajectories', so band_null_from_batch means what it did before
        drawn = self._dagger_expert.draw_bank_paths(
            self._plan_bank_labels(plan),
            null_classes=labels.tolist() if self.band_null_from_batch else None)
        if self.band_states == 'onpath':
            self._place_onpath_images(plan, drawn)
        return drawn

    def _place_onpath_images(self, plan, drawn):
        """Choose each onpath row's image at DRAW time, before any IO: a uniform image
        of the row's class bank in a uniform orientation, recorded as its bank entry
        ``plan['x0_entry'][r]`` (the bank's rows follow ``cond_paths`` order, and
        flips sit at ``+ n``). Under ``band_onpath_keep`` the image is also written
        into the row's NULL bank -- replacing one random draw, so the bank keeps its
        size -- so both posteriors contain it, as the full empirical distribution
        does; otherwise it is in the null bank only by chance (~null_bank_size / 1.28M)."""
        n_orient = 2 if self._dagger_expert.include_flips else 1
        entries, taken = [], {}
        for r in range(len(plan['traj'])):
            cond = drawn['cond_paths'][drawn['cond_index'][r]]
            j = int(torch.randint(len(cond), ()))
            entries.append(int(torch.randint(n_orient, ())) * len(cond) + j)
            if self.band_onpath_keep:
                null = drawn['null_paths'][drawn['null_index'][r]]
                if cond[j] not in null:
                    # never overwrite an image an earlier row placed (shared null bank)
                    used = taken.setdefault(id(null), set())
                    free = [i for i in range(len(null)) if i not in used]
                    i = free[int(torch.randint(len(free), ()))]
                    null[i] = cond[j]
                    used.add(i)
        plan['x0_entry'] = entries

    def _prefetch_banks(self, plan):
        """Issue the bank IO BEFORE the flow-matching step, mirroring the MMD target
        prefetch. Only the trajectories in ``plan['traj']`` -- the ones some scored
        row actually reads -- get banks.

        Bank construction is 85% image loading (measured: 3.51s of 4.13s per
        trajectory), and none of it depends on anything the rollout produces -- only
        on the labels, known here. Issuing it now gives the reads the whole FM
        forward/backward plus the rollout to finish in, so the cost leaves the step
        time instead of merely shrinking.

        Only the IO runs on the thread: paths are drawn and decoded to uint8 CPU
        tensors, and the encode/feat_fn half stays on the main thread at consumption,
        so no CUDA call is ever made off-thread. The null draw stays INDEPENDENT per
        trajectory -- this changes when the loading happens, not what is loaded.
        """
        expert = self._dagger_expert
        if expert is None or not expert.ready or not self._needs_banks():
            return
        if self._bank_pool is None:
            self._bank_pool = ThreadPoolExecutor(max_workers=1)
        drawn = self._draw_plan_paths(plan)
        # the reusable (pinned) host buffer is leased HERE, on the main thread, so
        # the prefetch thread only ever writes host memory; None -> fresh memory
        lease = expert.lease_host_buffer(drawn)
        self._bank_pending = self._bank_pool.submit(
            self._timed_load, expert, drawn, time.perf_counter(), lease)
        # whoever retires this draw releases the lease: the band that consumes it
        # (_release_band_lease) or _drop_bank_pending if it is abandoned
        self._bank_pending.bank_lease = lease

    @staticmethod
    def _timed_load(expert, drawn, t_issue, lease=None):
        """``expert.load_bank_images`` stamped with when the draw was issued and how
        long the load itself took, so consumption can tell whether the IO is hidden
        (``band_t_load`` < ``band_t_window``) and by how much. Pure host code: safe on
        the prefetch thread."""
        t0 = time.perf_counter()
        loaded = expert.load_bank_images(drawn, lease=lease)
        loaded['t_issue'] = t_issue
        loaded['t_load'] = time.perf_counter() - t0
        return loaded

    @staticmethod
    def _synced_clock(device):
        """``perf_counter`` once the device's queued work has finished, so the span
        between two calls is the GPU time it covers rather than its launch time."""
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        return time.perf_counter()

    @staticmethod
    def _plan_bank_labels(plan):
        """Labels of the trajectories that get banks, in ``plan['traj']`` order."""
        labels = plan['labels']
        return labels[torch.as_tensor(plan['traj'], device=labels.device)]

    def _drop_bank_pending(self):
        """Discard an unconsumed bank draw (claimed iteration that then bailed out)."""
        if self._bank_pending is not None:
            lease = getattr(self._bank_pending, 'bank_lease', None)
            if lease is not None:
                # cancel() cannot stop a load already writing the buffer, so the
                # lease is released only once the future is DONE: at once if the
                # cancel lands before the load starts, else when the load finishes.
                # Until then the buffer stays busy and is never handed out again.
                self._bank_pending.add_done_callback(
                    lambda _f, l=lease: self._dagger_expert.release_host_buffer(l))
            self._bank_pending.cancel()
            self._bank_pending = None

    def _release_band_lease(self):
        """Release the consumed band's host buffer. Called only after the band's
        closing device sync, when no upload can still be reading it."""
        lease, self._band_lease = getattr(self, '_band_lease', None), None
        if lease is not None:
            self._dagger_expert.release_host_buffer(lease)

    def _build_gap_banks(self, plan, encode_fn, device):
        """One :meth:`EmpiricalExpert.build_banks` call per iteration, shared by every
        scored state (the banks depend on the row's class, not on sigma). Returns
        banks for ``plan['traj']`` only, in that order.

        This is the expensive part of the arm: ``(band_rows + start rows) *
        (bank_size + null_bank_size)`` images are loaded and encoded, every iteration
        the band runs. Budget it with ``band_rows``, the expert's ``bank_size`` /
        ``null_bank_size``, and ``mmd_interval`` -- see the config's cost note.
        """
        expert = self._dagger_expert
        assert expert is not None and expert.ready, (
            'the empirical expert is required whenever a term reads v* -- the '
            'empirical FM term always does, and the CFG-gap term does unless it is '
            "running cfg_gap_target='zero'. v* is an average over per-class image "
            'banks, so set model.expert in the config (the wrapper hands the '
            "diffusion its handle). For a fully expert-free arm set emp_fm_weight=0 "
            "AND cfg_gap_target='zero'.")
        assert encode_fn is not None, (
            'the expert needs an encode_fn (images -> diffusion input space) to build '
            'its banks; the wrapper supplies it as expert_encode_fn. Got None.')
        t_consume = time.perf_counter()
        if self._bank_pending is not None:
            loaded = self._bank_pending.result()      # in flight since forward_train
            self._bank_pending = None
        else:                                         # fallback: synchronous, IO inline
            loaded = self._timed_load(expert, self._draw_plan_paths(plan), t_consume)
        t_ready = time.perf_counter()
        self._band_lease = loaded.get('lease')    # released at the end of this band
        banks = expert.finish_banks(loaded, encode_fn, self.feat_fn, device)
        self._log_bank_cost(loaded, banks, t_consume, t_ready,
                            self._synced_clock(device), device)
        return banks

    def _log_bank_cost(self, loaded, banks, t_consume, t_ready, t_done, device):
        """What this band's expert banks cost, and the largest NULL bank (images per
        trajectory) each resource would allow at the current settings -- logged so
        the ceiling on null_bank_size is read off a run instead of guessed.

        Timings (s): ``band_t_wait`` the main thread blocked on the IO (~0 when the
        prefetch hides it), ``band_t_load`` the IO itself on the prefetch thread,
        ``band_t_window`` how long the IO had between issue and use, ``band_t_finish``
        the GPU half (upload + feats). Memory (GiB, this rank): ``band_mem_gpu_bank``
        the resident bank feats, ``band_mem_host_bank`` the banks' host copies,
        ``band_mem_gpu_free`` device total minus the PEAK reserved so far,
        ``band_mem_host_avail`` the node's MemAvailable.

        Estimates (null images per trajectory, everything else held fixed), each
        linear in what one more null image costs:
          * ``null_bank_max_gpu``    -- the GPU headroom over its resident feats (both
            orientations, every null bank) plus x0_hat's two transient [M, Df]
            temporaries for one row. Conservative: it assumes the banks coexist with
            the all-time peak, which they may not.
          * ``null_bank_max_host``   -- the node's available RAM over its host copy on
            every local rank: in the reused buffer (with its growth headroom) when
            the expert leases one, else twice (the bank in use and the next band's,
            allocated fresh while it is still alive).
          * ``null_bank_max_hidden`` -- the largest bank whose load still fits the
            prefetch window, taking load time proportional to images loaded. Above it
            the IO stops being free and ``band_t_wait`` turns positive.
        """
        from .experts.empirical_expert import HostBankBuffer, U8Bank
        bt = self._band_t
        bt['band_t_wait'] = t_ready - t_consume
        bt['band_t_load'] = loaded['t_load']
        bt['band_t_window'] = t_consume - loaded['t_issue']
        bt['band_t_finish'] = t_done - t_ready

        def distinct(bank_list):     # aliased (deduped / shared) banks are held once
            return list({id(b): b for b in bank_list}.values())

        def feats(b):
            return b.feats if isinstance(b, U8Bank) else b[1]

        def host_bytes(b):
            return b.u8.nbytes if isinstance(b, U8Bank) else b[0].nbytes

        gib = float(2 ** 30)
        cond_b, null_b = distinct(banks[0]), distinct(banks[1])
        bt['band_mem_gpu_bank'] = sum(feats(b).nbytes for b in cond_b + null_b) / gib
        bt['band_mem_host_bank'] = sum(host_bytes(b) for b in cond_b + null_b) / gib
        if device.type == 'cuda':
            gpu_free = (torch.cuda.get_device_properties(device).total_memory
                        - torch.cuda.max_memory_reserved(device))
        else:
            gpu_free = 0
        host_avail = 0
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        host_avail = int(line.split()[1]) * 1024
                        break
        except OSError:
            pass
        bt['band_mem_gpu_free'] = gpu_free / gib
        bt['band_mem_host_avail'] = host_avail / gib
        bt['band_mem_host_buffers'] = self._dagger_expert.host_buffer_bytes / gib

        null_u8 = loaded['null_u8']
        n_null = null_u8[0].shape[0] if null_u8 else 0   # images per null bank
        bt['band_null_imgs'] = float(n_null)
        if not null_b or n_null == 0:
            return
        # per extra null image (one more per bank, in every distinct null bank)
        f0 = feats(null_b[0])
        per_entry = f0[0].nbytes                          # one entry's feats, bytes
        n_orient = f0.shape[0] // n_null
        gpu_per_img = len(null_b) * n_orient * per_entry + 2 * n_orient * per_entry
        copies = HostBankBuffer.HEADROOM if loaded.get('lease') is not None else 2
        host_per_img = (len(null_u8) * null_u8[0][0].nbytes * copies
                        * int(os.environ.get('LOCAL_WORLD_SIZE', 1)))
        bt['null_bank_max_gpu'] = n_null + gpu_free / gpu_per_img
        bt['null_bank_max_host'] = n_null + host_avail / host_per_img
        n_cond_imgs = sum(u.shape[0] for u in loaded['cond_u8'])
        n_null_imgs = sum(u.shape[0] for u in null_u8)
        if loaded['t_load'] > 0:
            per_img_load = loaded['t_load'] / max(n_cond_imgs + n_null_imgs, 1)
            fits = bt['band_t_window'] / per_img_load      # images loadable in the window
            bt['null_bank_max_hidden'] = max(fits - n_cond_imgs, 0.0) / len(null_u8)

    @torch.no_grad()
    def _onpath_band_states(self, plan, banks, x_ref):
        """``band_states='onpath'``: one noised REAL image per row, and the entries
        its conditional posterior leaves out. Row ``r``'s image is the class-bank
        entry fixed at draw time (:meth:`_place_onpath_images`: a uniform image of
        the class, either orientation), at a sigma from the training timestep
        sampler truncated to sigma >= the band edge (_sample_t_above_split, the draw
        reg_ft's own carve uses), so the band's states and sigma marginal are
        reg_ft's and only the target changes. Unless ``band_onpath_keep``, both
        orientations of the image are excluded, or the posterior would find x0
        itself and hand back eps - x0. Returns ``(x_t, sigma, exclude)``."""
        from .experts.empirical_expert import U8Bank
        assert banks is not None, (
            "band_states='onpath' reads its images out of the expert's class banks; "
            'no term that builds banks is on.')
        device = x_ref.device
        x0, exclude = [], []
        for r, k in enumerate(plan['x0_entry']):
            bank = banks[0][r]          # plan['traj'] is range(n_rows)
            assert isinstance(bank, U8Bank), (
                "band_states='onpath' needs the expert's bank_storage='u8'.")
            x0.append(bank.entry(k, device))
            exclude.append(None if self.band_onpath_keep else bank.orientations(k))
        x0 = torch.stack(x0).to(x_ref.dtype)
        t = self._sample_t_above_split(x0.size(0), x0.shape[2:].numel(), device)
        t = t.clamp(max=self.num_timesteps)
        x_t, _, _ = self.sample_forward_diffusion(x0, t, torch.randn_like(x0))
        return x_t, t / self.num_timesteps, exclude

    def _onpolicy_loss(self, x_0, plan, t_split, encode_fn=None):
        """One shared band rollout; the CFG-gap and (optional) MMD terms on top, and
        emp_fm's rows as a SUM over the micro-batch ``x_0`` -- they are the expert
        share of the shared FM mean (see :meth:`forward_train`).
        ``plan`` (from :meth:`_draw_band_plan`) fixes the classes and scored pairs."""
        self._band_t = {}
        t_band = self._synced_clock(x_0.device)
        n_batch = x_0.size(0)   # the shared FM mean's denominator
        labels = plan['labels']
        m = labels.numel()
        x_0 = x_0[:m].detach()
        # only the first n_roll trajectories are rolled out; the rest carry a t=1 row
        # alone (see _draw_band_plan)
        n_roll = plan['n_roll']

        use_mmd = self.mmd_weight != 0
        use_fm = self.emp_fm_weight != 0
        use_gap = self.cfg_gap_weight != 0
        use_point = use_gap or use_fm  # terms sharing the per-state forward
        # The trajectory graph exists only for MMD: the CFG-gap term reads detached
        # states, so with MMD off the rollout is pure inference and costs no
        # activations. See the class docstring.
        ctx = contextlib.nullcontext() if use_mmd else torch.no_grad()
        # band_score_start: the point terms also score the t=1 start state. It is split
        # off here, so `states` stays the landing states the plan and MMD index.
        with_start = self.band_score_start and use_point and plan.get('j_start')
        onpath = self.band_states == 'onpath'
        if onpath:
            states, n_roll = [], 0    # no rollout: the rows are built from the banks below
        else:
            with ctx:
                states = self._band_rollout(x_0[:n_roll], labels[:n_roll], t_split,
                                            include_start=bool(with_start))
        start = None
        if with_start:
            start, states = states[0], states[1:]
            if n_roll < m:
                # the start-only trajectories: noise drawn exactly as _band_rollout
                # draws its start, never stepped
                sampler = self._build_mmd_sampler(x_0.shape[2:].numel(), x_0.device)
                noise = (sampler.timesteps[0] / self.num_timesteps) \
                    * torch.randn_like(x_0[n_roll:])
                start = (start[0], torch.cat([start[1], noise]))

        # ONE band state per trajectory, drawn uniformly -- so the band contributes
        # exactly m rows with the band's own t-marginal, and the point terms are an
        # estimator of the band's objective rather than n_states copies of it. MMD is
        # unaffected below: it is a set statistic and still uses every state.
        pick = None
        if use_point and states:
            # ONE row per trajectory (each costs a bank), at a band state drawn
            # uniformly, so the band's t-marginal is uniform over the grid. The pairs
            # were drawn in _draw_band_plan, before the rollout, so the banks could be
            # loaded for the scored trajectories alone.
            assert len(states) == plan['n_states'], (
                f"band rollout returned {len(states)} states but the plan was drawn "
                f"over {plan['n_states']}; _n_band_states is out of sync with "
                '_band_rollout.')
            j_pick, k_pick = list(plan['j_pick']), plan['k_pick']
            xs = [states[k][1][j].detach() for j, k in zip(j_pick, k_pick)]
            sigs = [states[k][0] for k in k_pick]
            # each grid row carries its cell's mass, not the weight at its state
            row_w = self._band_grid(x_0, t_split)[1][k_pick]
            land_rows = None
            if start is not None:
                # t=1 rows appended after the landing ones (land_rows marks the latter,
                # for logging the two apart, and keeps them out of emp_fm's FM rows);
                # every other point term scores all rows
                n_land = len(j_pick)
                xs += [start[1][j].detach() for j in plan['j_start']]
                sigs += [start[0]] * len(plan['j_start'])
                j_pick += plan['j_start']
                row_w = torch.cat([row_w, row_w.new_ones(len(plan['j_start']))])
                land_rows = torch.arange(len(j_pick), device=x_0.device) < n_land
            x_pick = torch.stack(xs)
            sig_pick = x_0.new_tensor(sigs)
            lab_pick = labels[torch.as_tensor(j_pick, device=labels.device)]
            pick = (x_pick, sig_pick, lab_pick, j_pick, land_rows, row_w)

        # One bank set per iteration, reused by every scored state, for the scored
        # trajectories only (plan['traj']).
        banks = None
        if self._needs_banks():
            banks = self._build_gap_banks(plan, encode_fn, x_0.device)
        exclude = None
        if onpath and use_point:
            x_pick, sig_pick, exclude = self._onpath_band_states(plan, banks, x_0)
            # continuous sigma, as reg_ft draws it: the weight at the row is exact
            pick = (x_pick, sig_pick, labels, list(range(m)), None, None)

        gap_sigmas, gap_vals, fm_vals = [], [], []
        frac_vals = []
        sub_vals, comp_vals, empsub_vals, t1_vals, land_vals, raw_vals = [], [], [], [], [], []
        if pick is not None:
            # banks are indexed by position in plan['traj'], so re-index them onto the
            # drawn rows: row i scores against trajectory j_pick[i]'s banks.
            banks_pick = banks
            if banks is not None:
                pos = {j: i for i, j in enumerate(plan['traj'])}
                banks_pick = ([banks[0][pos[j]] for j in pick[3]],
                              [banks[1][pos[j]] for j in pick[3]])
            gap, fm_rows, fmdiag, gsplit = self._band_point_losses(
                pick[0], pick[1], pick[2], banks=banks_pick, exclude=exclude,
                row_w=pick[5])
            gap_sigmas.append(float(pick[1].mean()))
            if True:
                if gap is not None:
                    gap_vals.append(gap)
                    sub_vals.append(gsplit[0]); comp_vals.append(gsplit[1])
                    raw_vals.append(gsplit[3])
                    if pick[4] is not None:   # the gap at t=1 vs at the landing states
                        t1_vals.append(gsplit[2][~pick[4]].mean())
                        land_vals.append(gsplit[2][pick[4]].mean())
                if fm_rows is not None:
                    # the t=1 start rows are not FM rows: the carve made room for
                    # the landing rows only (and their weight is ~1e-52 anyway)
                    fm_vals.append(fm_rows if pick[4] is None else fm_rows[pick[4]])
                    frac_vals.append(fmdiag[0]); empsub_vals.append(fmdiag[1])

        mmd_loss = mmd_log_vars = None
        if use_mmd:
            # The target is class-matched to the BAND's labels, the ones the rollout
            # was generated under. The micro-batch rows qualify only under
            # band_class_sampling='batch', where the band reuses their labels; the
            # independent 'uniform'/'prior' draws take their targets from disk by
            # label (GaussianFlowMMD._target_per_row).
            mmd_loss, mmd_log_vars = self._mmd_score(
                states, x_0[:n_roll], labels[:n_roll],
                rows_match=self.band_class_sampling == 'batch')

        # *mmd_accum_steps cancels train_grad_accum's 1/N (both terms run on ONE
        # micro-batch, not all N). The extra *world_size applies to the MMD term
        # ALONE: its pooled estimate leaves each rank differentiating only its own
        # rows, so the per-rank gradients SUM to the true gradient and DDP's
        # averaging would shrink it by 1/W. The CFG-gap term is an ordinary
        # per-point mean -- every rank computes the same quantity over its own
        # rows -- so DDP averaging is already correct and it must NOT be rescaled.
        # emp_fm takes NO *acc at all: its rows are part of this micro-batch's
        # shared FM mean, which train_grad_accum's 1/N normalises like any other.
        acc = float(self.mmd_accum_steps)
        loss = x_0.new_zeros(())
        log_vars = dict(band_steps=loss.new_tensor(float(len(states)) * acc),
                        band_traj=loss.new_tensor(float(n_roll)),
                        band_banks=loss.new_tensor(
                            float(len(plan['traj'])) if banks is not None else 0.0),
                        band_rows=loss.new_tensor(
                            float(pick[0].shape[0]) if pick is not None else 0.0))

        if gap_vals:
            gap = torch.stack(gap_vals).mean()
            loss = loss + self.cfg_gap_weight * gap * acc
            log_vars['cfg_gap'] = gap.detach() * acc
            log_vars['cfg_gap_states'] = loss.new_tensor(float(len(gap_vals)) * acc)
            for sigma, v in zip(gap_sigmas, gap_vals):
                log_vars[f'cfg_gap_s{sigma:.3f}'] = v.detach() * acc
            log_vars['loss_cfg_gap'] = (self.cfg_gap_weight * gap).detach() * acc
            log_vars['w_cfg'] = loss.new_tensor(self.cfg_gap_weight * acc)
            # where the gap's energy sits: subspace is 8/768 = 1.04% of the dims, so
            # cfg_gap_sub >> 0.0104 * cfg_gap means class structure concentrates in
            # AsymJiT's basis and a projected target would be worth testing.
            gs = torch.stack(sub_vals).mean(); gc = torch.stack(comp_vals).mean()
            log_vars['cfg_gap_sub'] = gs * acc
            log_vars['cfg_gap_comp'] = gc * acc
            log_vars['cfg_gap_subfrac'] = gs / (gs + gc).clamp_min(1e-12) * acc
            if self.cfg_gap_reweight:   # the unweighted gap, comparable to plain-mean runs
                log_vars['cfg_gap_raw'] = torch.stack(raw_vals).mean() * acc
            if t1_vals:   # band_score_start: the t=1 rows' share of the gap, kept apart
                log_vars['cfg_gap_t1'] = torch.stack(t1_vals).mean() * acc
                log_vars['cfg_gap_land'] = torch.stack(land_vals).mean() * acc
                log_vars['band_rows_t1'] = loss.new_tensor(float((~pick[4]).sum()) * acc)
        if fm_vals:
            fm_rows = torch.cat(fm_vals)
            assert fm_rows.numel() == plan['n_roll'], (
                f"emp_fm scored {fm_rows.numel()} rows but the carve made room for "
                f"{plan['n_roll']}.")
            # the expert share of the shared FM mean: summed over its rows, divided
            # by the micro-batch like the on-path rows it replaced
            fm_term = self.emp_fm_weight * fm_rows.sum() / n_batch
            loss = loss + fm_term
            # per-row mean, comparable to the DAGGER arms' loss_dagger
            fm = fm_rows.mean()
            log_vars['emp_fm'] = fm.detach() * acc
            for sigma in gap_sigmas:
                log_vars[f'emp_fm_s{sigma:.3f}'] = fm.detach() * acc
            # its contribution to the step's loss (train_grad_accum's 1/N included)
            log_vars['loss_emp_fm'] = fm_term.detach()
            log_vars['band_fm_rows'] = loss.new_tensor(float(fm_rows.numel()) * acc)
            # realised conditional fraction -- should sit at band_prob_class (0.9)
            log_vars['fm_cond_frac'] = torch.stack(frac_vals).mean() * acc
            log_vars['emp_fm_subfrac'] = torch.stack(empsub_vals).mean() * acc
        if mmd_loss is not None:
            # already carries *mmd_accum_steps and the pooled *world_size (see
            # GaussianFlowMMD._mmd_score); its logged values are the true MMD^2
            loss = loss + self.mmd_weight * mmd_loss
            log_vars.update(mmd_log_vars)
            tags = ('mmd_sub', 'mmd_raw') if self.mmd_feature == 'both' else (
                'mmd_sub' if self.mmd_feature == 'subspace' else 'mmd_raw', )
            log_vars['loss_mmd'] = self.mmd_weight * sum(mmd_log_vars[t] for t in tags)
        # the band's forward wall time (its backward runs later, fused with the FM
        # term's) and the bank costs gathered on the way -- see _log_bank_cost
        self._band_t['band_t_total'] = self._synced_clock(x_0.device) - t_band
        # the sync above also retired every upload that read the banks' host buffer
        self._release_band_lease()
        for k, v in self._band_t.items():
            log_vars[k] = loss.new_tensor(float(v) * acc)
        return loss, log_vars

    def forward_train(
            self,
            x_0,
            visual_encoder_features=None,
            running_status=None,
            buffer_batch=None,
            n_onpath_high=0,
            class_labels_true=None,
            expert_encode_fn=None,
            **kwargs):
        # the MMD term's disk-drawn targets need the encoder too, and the truncated
        # on-path branch below never reaches GaussianFlowMMD.forward_train to set it
        if expert_encode_fn is not None:
            self._encode_fn = expert_encode_fn
        # A replay buffer is allowed ONLY when it feeds the inherited DAGGER stream
        # while the band terms stay online. The band's own points always come from a
        # rollout taken with the CURRENT weights -- never from the buffer -- so the
        # CFG-gap/MMD terms remain strictly on-policy even here. What the
        # buffer must not do is supply those band points, which it cannot: it is
        # consumed only by GaussianFlowDagger.forward_train below.
        # Claim + prefetch BEFORE the flow-matching step: the claim decides which
        # micro-batch runs the band, and issuing the bank IO here buys it the whole
        # FM forward/backward plus the rollout to complete in.
        t_split_pre = self.mmd_t_split if self.mmd_t_split is not None else self.t_split
        band_labels = class_labels_true if class_labels_true is not None \
            else kwargs.get('class_labels', None)
        band_ok = t_split_pre is not None and band_labels is not None
        run_band = self._band_claim(running_status) and band_ok
        # every micro-batch of a band iteration, not only the one that runs the band:
        # all of them take the carve's on-path split (see _carve_counts)
        band_iter = band_ok and self._band_due(running_status)
        plan = None
        if run_band:
            if self._bank_ready is not None:
                # DOUBLE BUFFER: this draw was issued when the PREVIOUS band ran, so
                # it has had the whole inter-band interval (mmd_interval iterations)
                # to load rather than a single FM step.
                plan, self._bank_pending = self._bank_ready
                self._bank_ready = None
            else:
                # first band of the run (or after an interval change): no buffer yet
                plan = self._draw_band_plan(x_0, t_split_pre, band_labels)
                self._prefetch_banks(plan)

        assert buffer_batch is None or not self._truncate_onpath(), (
            'a DAGGER buffer and an on-path truncation cannot both be active: the '
            'buffer carve already reserves the high-sigma rows, so truncating on top '
            'would leave the band supervised twice. Set emp_fm_weight=0 to let the '
            'DAGGER stream own the band, or drop the DaggerRolloutHook.')

        # On-path flow matching. In a band iteration with the on-path truncated (i.e.
        # emp_fm on), THE CARVE, as in d-flow and DAGGER's proportional carve: the FM
        # loss is ONE mean over this micro-batch, the band's expert rows (emp_fm,
        # summed in _onpolicy_loss) take the place of as many on-path rows, and the
        # on-path rows left stop at the band edge apart from band_frac_on_path's
        # real-data share of it. Every FM row of the step then weighs 1/batch, so the
        # row counts alone carry reg_ft's sigma allocation. Otherwise -- no band this
        # iteration (warmup, interval) or no emp_fm -- full-range, exactly as reg_ft.
        if self._truncate_onpath() and band_iter:
            n_batch = x_0.size(0)
            n_expert = plan['n_roll'] if (run_band and self.emp_fm_weight != 0) else 0
            n_on, n_above = self._carve_counts(n_batch, n_expert)
            kw_on = dict(kwargs)
            if torch.is_tensor(kw_on.get('class_labels', None)):
                kw_on['class_labels'] = kw_on['class_labels'][:n_on]
            loss, log_vars = self._onpath_loss(
                x_0[:n_on], truncate=True, use_expert=False, n_above=n_above, **kw_on)
            loss = loss * (n_on / n_batch)   # its rows' share of the micro-batch mean
            log_vars['onpath_rows'] = loss.new_tensor(float(n_on))
            log_vars['onpath_n_above'] = loss.new_tensor(float(n_above))
        else:
            # forward the DAGGER buffer through: GaussianFlowDagger.forward_train
            # mixes its expert-labelled rollout term in as the usual convex
            # combination, leaving that objective bit-identical to the plain arm.
            loss, log_vars = super().forward_train(
                x_0,
                visual_encoder_features=visual_encoder_features,
                running_status=running_status,
                buffer_batch=buffer_batch,
                n_onpath_high=n_onpath_high,
                **kwargs)

        t_split = self.mmd_t_split if self.mmd_t_split is not None else self.t_split
        # the CFG-gap term needs the TRUE class labels: kwargs['class_labels'] has
        # already had CFG dropout applied, and a row relabelled to null would compare
        # the unconditional branch against itself and contribute an exact zero. The
        # wrapper supplies the undropped labels; fall back to the dropout-applied
        # ones if it did not (the term then just loses those rows' signal).
        if run_band:
            band_loss, band_log_vars = self._onpolicy_loss(
                x_0, plan, t_split, encode_fn=expert_encode_fn)
            loss = loss + band_loss
            log_vars.update(band_log_vars)
            self._drop_bank_pending()   # this band's draw is spent
            # Queue the NEXT band's banks now. Only possible because the classes are
            # drawn independently of the minibatch: they do not wait on a future
            # batch, so the load gets the entire interval to finish in.
            if self.band_class_sampling != 'batch' and self._dagger_expert is not None:
                nxt = self._draw_band_plan(x_0, t_split)
                self._prefetch_banks(nxt)
                self._bank_ready = (nxt, self._bank_pending)
                self._bank_pending = None
        else:
            self._drop_bank_pending()

        return loss, log_vars
