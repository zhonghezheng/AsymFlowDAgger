# Copyright (c) 2026 Hansheng Chen

import contextlib
import inspect

import os.path as osp
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from lakonlab.datasets.imagenet import image_preproc
import torch.distributed as dist
import diffusers

from ..builder import MODULES
from . import schedulers
from .gaussian_flow_dagger import GaussianFlowDagger


def _pdist2(x, y):
    """Squared Euclidean distance matrix: ``[m, D] x [n, D] -> [m, n]``."""
    x2 = x.pow(2).sum(-1)
    y2 = y.pow(2).sum(-1)
    return (x2.unsqueeze(1) + y2.unsqueeze(0) - 2.0 * (x @ y.transpose(0, 1))).clamp_min(0)


def mmd2_rbf(fx, fy, bandwidths=(0.25, 0.5, 1.0, 2.0, 4.0), unbiased=True, eps=1e-12):
    """Multi-bandwidth RBF MMD^2 between a MODEL sample set ``fx`` (carries grad) and
    a TARGET set ``fy`` (detached).

        MMD^2_u = mean_{i!=j} k(xi,xj) + mean_{i!=j} k(yi,yj) - 2 mean k(xi,yj)
        k(a,b)  = mean_l exp(-||a-b||^2 / (scale_l * d2_base))

    ``d2_base`` is the MEAN SQUARED DISTANCE BETWEEN TWO DISTINCT TARGET SAMPLES,
    measured on ``fy`` alone -- d-flow's rule (cifar_fm_dagger_conditional_mmd.py
    :func:`mmd2_unbiased`). Taking the width from the data side keeps the kernel
    theta-INDEPENDENT: it is a fixed measuring stick rather than a statistic that
    drifts as the policy moves. It is also t-adaptive for free -- at sigma=1 both
    sets are unit noise and d2_base = 2*D exactly, and as sigma falls the data term
    takes over -- so one set of multipliers spans the whole band without retuning.
    The diagonal of d2_yy is identically zero, hence /(n(n-1)) rather than /n^2;
    including it would drag the width down by (n-1)/n.

    NB this replaced an earlier base -- the MEDIAN of the CROSS distances, divided by
    the feature dimension, with an extra factor 2 in the denominator. Measured, the
    two agree to ~3% over a 1.6-magnitude drift of the model away from the data: the
    16x span of the multiplier mixture absorbs a 2-3x shift in the base, since the
    shift only re-indexes which multiplier carries the signal. So this is a change of
    PRINCIPLE (theta-independence, and parity with the reference implementation), not
    of magnitude, and earlier arms' numbers remain comparable.

    ``unbiased`` drops the diagonal from the two within-set terms (the standard
    unbiased MMD^2 estimator, which can be slightly negative when the two
    distributions coincide -- expected, and informative in the log).
    """
    fx = fx.float()
    fy = fy.float()
    m, n = fx.shape[0], fy.shape[0]
    assert m > 1 and n > 1, (
        f'MMD^2 needs at least 2 samples per side, got m={m}, n={n}. Raise '
        f'mmd_batch or mmd_target_n.')
    d_xx = _pdist2(fx, fx)
    d_yy = _pdist2(fy, fy)
    d_xy = _pdist2(fx, fy)
    with torch.no_grad():
        base = (d_yy.detach().sum() / (n * (n - 1))).clamp_min(eps)
    total = 0.0
    for mult in bandwidths:
        denom = base * mult
        k_xx = torch.exp(-d_xx / denom)
        k_yy = torch.exp(-d_yy / denom)
        k_xy = torch.exp(-d_xy / denom)
        if unbiased:
            e_xx = (k_xx.sum() - k_xx.diagonal().sum()) / (m * (m - 1))
            e_yy = (k_yy.sum() - k_yy.diagonal().sum()) / (n * (n - 1))
        else:
            e_xx = k_xx.mean()
            e_yy = k_yy.mean()
        total = total + e_xx + e_yy - 2.0 * k_xy.mean()
    return total / len(bandwidths)


@MODULES.register_module()
class GaussianFlowMMD(GaussianFlowDagger):
    """GaussianFlow(+DAGGER) with an added per-NFE-step distribution-matching term.

    On top of the usual flow-matching loss, every train iteration rolls a small
    batch out from noise with the *current* policy, using the eval sampler and NFE
    grid, and stops once the state leaves the ``mmd_t_split`` band. At every
    visited state inside the band it forms an MMD between

      * the ROLLOUT distribution   ``{x_sigma^i}`` (the policy's own marginal), and
      * the NOISED ON-PATH distribution ``{(1 - sigma) x_0^j + sigma eps^j}``
        (the forward-diffusion marginal at the same sigma, from the real latents
        already in the minibatch, fresh noise each step),

    and adds ``mmd_weight * mean_over_steps(MMD^2)`` to the loss. The rollout is
    built WITH gradient (backprop runs through every Heun step in the band), so
    the term trains the policy to make its own high-sigma marginals match the
    forward-diffusion marginals -- a distribution-level target instead of DAGGER's
    per-point expert velocity.

    Class matching: the rollout reuses the minibatch rows' (dropout-applied)
    ``class_labels`` and those same rows supply the on-path set, so the two sides
    share an identical class mixture and the MMD cannot be driven by class-mixture
    mismatch. By default the rollout is unguided (no CFG), matching inference in
    this band (the eval ``guidance_interval`` upper edge is 0.88); set
    ``mmd_guidance_scale`` to roll the source side out from the CFG-guided sampler
    instead.

    With ``expert=None`` and no ``DaggerRolloutHook`` the inherited DAGGER stream is
    inert, so the objective is exactly ``reg_ft + MMD`` (the ``regft_mmd`` arm).

    Args:
        mmd_weight (float): coefficient on the (mean-over-steps) MMD^2 term. 0
            disables the whole branch.
        mmd_t_split (float | None): band lower edge -- MMD is computed at every
            visited state with ``sigma >= mmd_t_split``. ``None`` falls back to
            ``self.t_split``. NB: this is independent of the FM carve, so the
            flow-matching loss can stay full-range (``t_split=None``).
        mmd_nfe (int): NFE of the rollout grid; match the eval sampler (50) so the
            visited states are the ones inference actually visits.
        mmd_batch (int): rollout samples per step (also the on-path sample count).
            Cost is ``2 * n_band_steps`` network evals at this batch size, with
            gradient; the MMD estimator improves with batch size.
        mmd_sampler (str): scheduler name; match the eval sampler.
        mmd_guidance_scale (float): CFG scale of the rollout (source) side. 1.0 =
            unguided (the default). Above 1, every rollout eval is one batched
            ``[null; cond]`` forward combined as ``forward_test`` combines it,
            ``u = u_c + (w - 1)(u_c - u_u)``, with gradient through BOTH branches.
            The target side is unchanged (noised real data), so the term trains
            the GUIDED marginal toward the data marginal. Each eval runs at twice
            the batch, so the band graph's memory roughly doubles.
        mmd_guidance_interval (tuple | None): ``[lo, hi]`` in model-timestep units
            (sigma, at num_timesteps=1); guidance applies only to evals with
            ``lo <= t <= hi``, exactly as ``forward_test`` gates it. ``None`` =
            guided at every eval, the no_grad prefix above ``mmd_t_hi`` included.
            NB the eval interval [0, 0.88] would leave a sigma >= 0.875 band almost
            entirely unguided.
        mmd_feature (str): ``'subspace'`` (the AsymJiT rank-``basis_rank``
            features, ``feat_fn``), ``'raw'`` (flattened latents), or ``'both'``
            (sum). The space(s) NOT trained on are still computed under no_grad
            and logged, so both are always visible.
        mmd_bandwidths (tuple): RBF bandwidth multipliers on the median heuristic.
        mmd_unbiased (bool): unbiased (diagonal-free) MMD^2 estimator.
        mmd_interval (int): compute the term every N iterations (1 = every step).
        mmd_start_iter (int): no MMD before this iteration (LR/optimizer warmup on
            pure flow matching first, like the DAGGER rounds).
        mmd_eval_mode (bool): put the denoising net in eval mode for the rollout
            (dropout off), so the sampled marginal matches inference. Gradient
            checkpointing is keyed on ``torch.is_grad_enabled()``, not on training
            mode, so it stays active either way.
        mmd_step_checkpoint (bool): recompute each solver eval in the backward pass
            instead of retaining it. The band graph is the memory bottleneck -- the
            net already checkpoints per block, so each eval still holds 32 block
            boundaries (~1.5 GB at mmd_batch=64), and 12 evals is ~19 GB retained on
            top of the bs=256 flow-matching graph, which does not fit on a 140 GB
            card. Wrapping each eval in its own checkpoint drops that to the eval
            INPUTS (~50 MB each) at the cost of one extra forward per eval. Exact,
            not an approximation: full-band backprop is unchanged.
        mmd_gather (bool): pool the band states across DDP ranks into a single
            MMD over ``world_size * mmd_batch`` trajectories (see the note in
            __init__). The gather is differentiable only through the local rows --
            the other ranks' rows are constants -- so each rank contributes
            d(MMD)/d(theta) through its own slice and DDP's gradient AVERAGING would
            otherwise shrink the pooled gradient by 1/world_size; the loss is scaled
            by world_size to undo that. Logged values stay the true (unscaled) MMD^2.
        mmd_grad_probe (bool): diagnostic -- log the parameter-gradient norm of the
            flow-matching term and of the (unweighted) MMD term separately, which is
            what ``mmd_weight`` actually has to be set against (the two loss VALUES
            live on different scales). Costs two extra backward passes per iteration;
            for smoke runs, not for training. Uses ``autograd.grad``, which does not
            touch ``.grad`` and so does not fire DDP's reducer hooks.
    """

    def __init__(self,
                 *args,
                 mmd_weight=1.0,
                 mmd_t_split=None,
                 mmd_t_hi=None,
                 mmd_target_n=None,
                 mmd_classes_per_batch=None,
                 mmd_target_root='data/imagenet/train/',
                 mmd_target_datalist='data/imagenet/train.txt',
                 mmd_target_workers=32,
                 mmd_target_u8_cache=None,
                 mmd_nfe=50,
                 mmd_batch=64,
                 mmd_sampler='FlowHeunODE',
                 mmd_guidance_scale=1.0,
                 mmd_guidance_interval=None,
                 mmd_feature='subspace',
                 mmd_bandwidths=(0.25, 0.5, 1.0, 2.0, 4.0),
                 mmd_unbiased=True,
                 mmd_interval=1,
                 mmd_start_iter=0,
                 mmd_eval_mode=True,
                 mmd_step_checkpoint=False,
                 mmd_accum_steps=1,
                 mmd_gather=True,
                 mmd_grad_probe=False,
                 mmd_target_share='none',
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert mmd_feature in ('subspace', 'raw', 'both')
        if mmd_feature in ('subspace', 'both'):
            assert hasattr(self.denoising, 'proj_buffer'), \
                "mmd_feature='subspace' needs an AsymJiT-style denoising with proj_buffer."
        self.mmd_weight = float(mmd_weight)
        # How much of the target draw is REUSED across the band's timesteps
        # (d-flow's mmd_target_share). Every setting is marginally correct at each
        # step -- (1-s)x_0 + s*eps has the right law for ANY fixed (x_0, eps) pair --
        # so each per-step MMD^2 stays unbiased. What moves is the CORRELATION
        # between steps, hence the variance of the mean over them and of its
        # gradient.
        #   'none'  fresh images AND fresh noise per step (the default, and what
        #           every arm before this flag ran).
        #   'bank'  images drawn once per iteration and reused at every step; noise
        #           redrawn per step. The steps share a data sample, not a noise one.
        #   'both'  images AND noise drawn once, so each target point traces the
        #           straight line (1-s)x_0 + s*eps across the band -- the target set
        #           becomes a set of TRAJECTORIES, mirroring the rollout side's
        #           structure instead of being independent per step.
        # A draw costs no network evaluation, so sharing buys variance structure,
        # not compute.
        assert mmd_target_share in ('none', 'bank', 'both'), \
            f"mmd_target_share must be 'none', 'bank' or 'both', got {mmd_target_share!r}"
        self.mmd_target_share = mmd_target_share
        self.mmd_t_split = mmd_t_split
        # Upper edge of the band WINDOW. None -> the band is [mmd_t_split, 1.0] and
        # the whole rollout carries gradient. Set it to make the band a window
        # [mmd_t_split, mmd_t_hi]: the trajectory above mmd_t_hi is integrated under
        # no_grad (inference only, no graph) and gradient flows ONLY through the
        # window's steps.
        #
        # Why a window is worth having: with x_t = (1-s)x_0 + s*eps the DATA-dependent
        # share of the variance is (1-s)^2 / ((1-s)^2 + s^2) -- 0.75% at s=0.92 but
        # 3% at 0.85 and 10% at 0.75. The s*eps term is identically distributed on
        # both sides of the MMD, so it contributes nothing to the discrepancy while
        # dominating the pairwise distances the median-heuristic bandwidth is set
        # from. Measured: a 10%-narrower generated distribution is INVISIBLE at
        # s>=0.88 (SNR ~ 0) and clearly visible once the noise is removed. Moving the
        # window down trades backprop depth for a measurable signal.
        self.mmd_t_hi = mmd_t_hi
        # POOLED number of target (noised on-path) samples per MMD estimate. None ->
        # the same rows as the rollout side (world_size * mmd_batch). MMD does not
        # require m == n, and the target side is far cheaper than the rollout side:
        # no network forward, no graph, no gradient.
        #
        # For mmd_feature='subspace' the extra samples are exact and nearly free.
        # feat_fn is LINEAR (patchify/pack are reshapes, then a matmul), so
        #     feat((1-s)x_0 + s*eps) = (1-s)feat(x_0) + s*feat(eps)
        # and proj_buffer is orthonormal, so feat(eps) is white in feature space.
        # The target side can therefore be generated entirely from a cache of past
        # feat(x_0) rows -- 2048-d each, ~8 KB, versus 786 KB for a latent -- with the
        # noise drawn directly in feature space. Both identities verified numerically.
        #
        # Gain is bounded: for P==Q the estimator variance keeps a 1/(m(m-1)) term
        # from the ROLLOUT side, so raising n alone saturates. Measured (m=256):
        # floor sd 1.4e-4 at n=256 -> 1.0e-4 at 1024 -> 8.5e-5 at 2048, flat after.
        # Worth ~1.5x in SNR; mmd_batch remains the binding constraint.
        # CLASS-MATCHED: the extra target rows are drawn from the SAME classes the
        # rollout used, never from the data pool at large. With a perfect model and a
        # target spanning all 1000 classes while the rollout covers ~64, MMD^2 reads
        # +1.0e-3 -- ten times the noise floor and ~40% of a real signal -- and it does
        # NOT shrink with n, because it is a class-coverage artifact of the batch that
        # the model cannot remove. Training against it would push each class to
        # broaden toward the full-class mixture. Measured; see mmd_target_n.
        # Null-labelled (CFG-dropout) rows are the exception: their rollout IS the
        # unconditional marginal, so their target is drawn from all classes.
        self.mmd_target_n = mmd_target_n
        # Confine each step's rollout to this many DISTINCT classes, so every class
        # gets several trajectories instead of one. With 1000 ImageNet classes a
        # 64-trajectory batch covers ~62 of them at ~1.03 trajectories each, and the
        # per-row class matching then makes the class composition identical on both
        # sides -- which cancels between-class variation and leaves the statistic
        # measuring WITHIN-class detail from a single rollout sample per class.
        # Restricting to C classes gives m/C trajectories each (d-flow's
        # classes_per_batch; it runs 64 trajectories over 10 CIFAR classes = 6.4 each).
        # None = off, the historical behaviour.
        self.mmd_classes_per_batch = mmd_classes_per_batch
        # Targets are drawn FRESH FROM DISK, from the class's ENTIRE pool, separately
        # at every band timestep -- not from a cache of rows the batch already held.
        # A per-class FIFO of past batch rows is only as deep as what training has
        # recently seen (32/class), so successive timesteps resample the same handful
        # of images and the target marginal is a small fixed set rather than the
        # class distribution. Reading from the datalist costs ~1 ms/image wall at 8-32
        # threads on this fileset, which the band can absorb.
        self.mmd_target_root = mmd_target_root
        self.mmd_target_datalist = mmd_target_datalist
        self.mmd_target_workers = int(mmd_target_workers)
        # optional preprocessed uint8 cache (tools/build_imagenet_u8_cache.py). The
        # cache stores unflipped images; the random horizontal flip the JPEG path
        # applies (image_preproc random_flip=True) is applied at read time instead,
        # per image with p=0.5, so the target marginal is unchanged.
        from lakonlab.datasets.u8_cache import U8ImageCache
        self._target_u8 = U8ImageCache(mmd_target_u8_cache) if mmd_target_u8_cache else None
        self._target_paths = None      # class id -> list of relpaths (lazy, from disk)
        self._target_pool = None       # ThreadPoolExecutor for image loading
        self._prefetch_pool = None     # single orchestrator thread issuing the draws
        self._pending = deque()        # in-flight draws, one per band timestep
        self._encode_fn = None         # images -> diffusion input space (from wrapper)
        self.null_label_for_target = 1000
        self.mmd_nfe = int(mmd_nfe)
        self.mmd_batch = int(mmd_batch)
        self.mmd_sampler = mmd_sampler
        assert mmd_guidance_scale >= 1.0, (
            f'mmd_guidance_scale must be >= 1 (1 = unguided), got {mmd_guidance_scale}')
        self.mmd_guidance_scale = float(mmd_guidance_scale)
        self.mmd_guidance_interval = None if mmd_guidance_interval is None \
            else tuple(float(v) for v in mmd_guidance_interval)
        self.mmd_feature = mmd_feature
        self.mmd_bandwidths = tuple(mmd_bandwidths)
        self.mmd_unbiased = mmd_unbiased
        self.mmd_interval = int(mmd_interval)
        self.mmd_start_iter = int(mmd_start_iter)
        self.mmd_eval_mode = mmd_eval_mode
        assert not (mmd_step_checkpoint
                    and getattr(self.denoising, 'gradient_checkpointing', False)), (
            'mmd_step_checkpoint cannot be combined with the denoising net\'s own '
            'per-block checkpointing: the inner non-reentrant checkpoints recompute '
            'lazily DURING the outer recompute, which injects extra tensors (e.g. '
            'dropout masks) into the outer frame and trips CheckpointError '
            "'recomputed values have different metadata'. Halve the flow-matching "
            'graph with train_cfg.grad_accum_batch_size instead (see '
            'mmd_accum_steps), or set denoising.checkpointing=False.')
        self.mmd_step_checkpoint = mmd_step_checkpoint
        # micro-batches per optimizer step (train_cfg.grad_accum_batch_size). The MMD
        # term runs on exactly ONE of them -- rolling out once per micro-batch would
        # multiply the expensive part by mmd_accum_steps for no statistical gain --
        # and is scaled by this factor to cancel train_grad_accum's 1/N averaging, so
        # both the gradient and the logged values are the same as with no accumulation.
        assert mmd_accum_steps >= 1
        self.mmd_accum_steps = int(mmd_accum_steps)
        # pool the band states across ranks so ONE MMD is taken over all
        # world_size * mmd_batch trajectories, instead of each rank estimating from
        # its own mmd_batch and DDP averaging the results. Same rollouts, same evals
        # -- only the kernel matrix grows -- but a tighter estimate: for P==Q the
        # unbiased MMD^2 estimator's sd falls like 1/m within one estimate, while
        # averaging W of them only falls like 1/sqrt(W). Measured (D=256): one m=256
        # has sd 1.6e-4 vs 2.8e-4 for the mean of four m=64 (and ~5.5e-4 for a single
        # m=64) -- so ~2x tighter than the averaged alternative, ~3.4x than one rank's.
        # Enough to make a 5% scale discrepancy marginally resolvable (3.0e-4 against
        # a 1.6e-4 floor) where four averaged m=64 estimates cannot see it at all.
        self.mmd_gather = mmd_gather
        self._mmd_done_iter = None  # per-iteration latch (which micro-batch got the MMD)
        assert not (mmd_grad_probe and mmd_step_checkpoint), (
            'mmd_grad_probe needs extra backward passes (retain_graph) through the '
            'band graph, which a non-reentrant checkpoint region does not support: '
            'its recomputed tensors are cleared after the first backward. Run the '
            'probe with mmd_step_checkpoint=False (and a small mmd_batch, or it will '
            'not fit).')
        self.mmd_grad_probe = mmd_grad_probe

    # ---- band rollout -------------------------------------------------------

    def _build_mmd_sampler(self, seq_len, device):
        """The eval scheduler, built exactly as ``forward_test`` builds it (same
        shift / dynamic-shifting settings), so the sigma grid is the inference grid."""
        sampler_class = getattr(diffusers.schedulers, self.mmd_sampler + 'Scheduler', None)
        if sampler_class is None:
            sampler_class = getattr(schedulers, self.mmd_sampler + 'Scheduler', None)
        if sampler_class is None:
            raise AttributeError(f'Cannot find sampler [{self.mmd_sampler}].')
        signatures = inspect.signature(sampler_class).parameters.keys()
        sampler_kwargs = dict()
        for key in ['shift', 'use_dynamic_shifting', 'base_seq_len', 'max_seq_len',
                    'base_logshift', 'max_logshift']:
            if key in signatures and hasattr(self.timestep_sampler, key):
                sampler_kwargs[key] = getattr(self.timestep_sampler, key)
        sampler = sampler_class(self.num_timesteps, **sampler_kwargs)
        if 'seq_len' in inspect.signature(sampler.set_timesteps).parameters.keys():
            sampler.set_timesteps(self.mmd_nfe, seq_len=seq_len, device=device)
        else:
            sampler.set_timesteps(self.mmd_nfe, device=device)
        return sampler

    def _ckpt_pred(self, x_t, t, class_labels):
        """One solver eval, recomputed in the backward pass (see
        ``mmd_step_checkpoint``). Only ``x_t`` needs to be a checkpoint input: ``t``
        and ``class_labels`` carry no gradient, and the parameters are handled the
        same way the net's own per-block checkpoints handle them.

        The eval-mode switch and the compile bypass live INSIDE the checkpointed
        callable, deliberately. A checkpoint's recompute runs during
        ``loss.backward()`` -- long after ``forward_train`` has returned -- so any
        module state set up around the rollout and restored in a ``finally`` is
        already back to normal by then, and the recompute silently runs a DIFFERENT
        function than the forward did. That shows up as CheckpointError
        'recomputed values have different metadata': eager tensors saved on the
        forward, inductor buffers on the recompute. Putting the state inside ``fn``
        makes every call -- forward and recompute alike -- identical.

        Why the two switches at all: dropout must be off so the recompute is exact
        (and so the rollout matches inference), and torch.compile must be off
        because dynamo does not guarantee the same traced graph, hence the same
        saved-tensor set, when it is re-entered from inside a backward pass. Only
        these band evals go eager; the bs=256 flow-matching step stays compiled.

        With ``mmd_guidance_scale > 1`` the eval is the CFG-guided velocity, and the
        guidance combination also lives inside ``fn`` so the recompute is the same
        guided function. Whether this ``t`` is guided is decided up front from the
        interval -- deterministic in ``t``, so forward and recompute agree.
        """
        ckpt = self.mmd_step_checkpoint and torch.is_grad_enabled()
        w = self.mmd_guidance_scale
        if w > 1.0 and self.mmd_guidance_interval is not None:
            lo, hi = self.mmd_guidance_interval
            # compared as a tensor, as forward_test does: float(t) would promote to
            # double and can disagree with it at an edge such as sigma=0.88
            if not (lo <= t <= hi):
                w = 1.0

        def fn(x):
            net = self.denoising
            was_training = net.training
            compiled = getattr(net, '_compiled_forward', None)
            drop_compile = ckpt and compiled is not None
            if self.mmd_eval_mode and was_training:
                net.eval()
            if drop_compile:
                net._compiled_forward = None
            try:
                if w == 1.0:
                    return self.pred(x, t, class_labels=class_labels)
                # one batched [null; cond] eval, combined as forward_test does at the
                # eval setting orthogonal_guidance=0 (guidance_jit reduces to this)
                null = torch.full_like(class_labels, int(net.num_classes))
                u_u, u_c = self.pred(
                    torch.cat([x, x], dim=0), t,
                    class_labels=torch.cat([null, class_labels], dim=0)).chunk(2, dim=0)
                return u_c + (u_c - u_u) * (w - 1.0)
            finally:
                if drop_compile:
                    net._compiled_forward = compiled
                if self.mmd_eval_mode and was_training:
                    net.train()

        if not ckpt:
            return fn(x_t)
        return torch.utils.checkpoint.checkpoint(fn, x_t, use_reentrant=False)

    def _band_rollout(self, x_ref, class_labels, t_split):
        """Roll out from noise with the current policy for exactly the solver steps
        that stay inside the band, keeping the graph. Returns ``[(sigma, x_sigma),
        ...]`` for the states INSIDE the band (the sigma=1 start is excluded: both
        distributions are exactly N(0, I) there, so its MMD is identically 0).

        The Heun scheduler is stepped exactly as in ``forward_test`` -- two network
        evals per step (predictor at sigma_k, corrector at sigma_{k+1}) -- so the
        visited states are the inference states, only differentiable. ``sigmas`` is
        the Heun-doubled grid ``[s0, s1, s1, s2, s2, ...]``, so the state after the
        m-th completed step sits at ``sigmas[2m]``; the band step count is read off
        that grid up front, rather than by taking a step and discovering it left the
        band (which would burn two evals and their activations for nothing).
        """
        device = x_ref.device
        seq_len = x_ref.shape[2:].numel()
        sampler = self._build_mmd_sampler(seq_len, device)
        timesteps = sampler.timesteps
        grid = sampler.sigmas

        # steps whose landing sigma is still ABOVE the window: integrated without
        # gradient, so the graph starts at the window's upper edge.
        n_pre = 0
        if self.mmd_t_hi is not None:
            while 2 * (n_pre + 1) < len(grid) \
                    and float(grid[2 * (n_pre + 1)]) > self.mmd_t_hi:
                n_pre += 1
        n_band = n_pre
        while 2 * (n_band + 1) < len(grid) \
                and float(grid[2 * (n_band + 1)]) >= t_split:
            n_band += 1
        assert n_band > 0, (
            f'mmd_t_split={t_split} leaves no solver step inside the band at '
            f'mmd_nfe={self.mmd_nfe} (first landing sigma is {float(grid[2]):.4f}).')

        x_t = (timesteps[0] / self.num_timesteps) * torch.randn_like(x_ref)
        states = []
        i = 0
        for step in range(n_band):
            # above the window -> no_grad, so x_t re-enters the loop detached and the
            # graph spans the window alone
            ctx = torch.no_grad() if step < n_pre else contextlib.nullcontext()
            with ctx:
                out = self._ckpt_pred(x_t, timesteps[i], class_labels)
                x_t = sampler.step(out, timesteps[i], x_t, return_dict=False)[0]
                i += 1
                if not sampler.state_in_first_order:  # Heun corrector at the landing
                    out = self._ckpt_pred(x_t, timesteps[i], class_labels)
                    x_t = sampler.step(out, timesteps[i], x_t, return_dict=False)[0]
                    i += 1
            if step >= n_pre:
                states.append((float(sampler.sigmas[sampler.step_index]), x_t))
        return states

    # ---- the MMD term -------------------------------------------------------

    def _gather_feats(self, f):
        """All-gather features along dim 0, keeping the gradient path for the LOCAL
        rows: ``all_gather`` is not differentiable, so the local rank's slot is
        overwritten with the graph-carrying tensor and the other ranks' rows enter as
        constants. Every rank ends up with the same pooled set, hence the same MMD.
        Returns ``(pooled, world_size)``."""
        if not self.mmd_gather or not (dist.is_available() and dist.is_initialized()):
            return f, 1
        world_size = dist.get_world_size()
        if world_size == 1:
            return f, 1
        buf = [torch.empty_like(f) for _ in range(world_size)]
        dist.all_gather(buf, f.detach().contiguous())
        buf[dist.get_rank()] = f
        return torch.cat(buf, dim=0), world_size

    @torch.no_grad()
    def _init_target_index(self):
        """Parse the datalist metadata (``<relpath> <label>`` per line) into per-class
        path lists, once. This is the full training set -- ~1.28M rows over 1000
        classes -- so a draw sees the whole class, not a recent slice of it."""
        if self._target_paths is not None:
            return
        paths = [[] for _ in range(self.denoising.num_classes)]
        with open(self.mmd_target_datalist) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                c = int(parts[1])
                if 0 <= c < len(paths):
                    paths[c].append(parts[0])
        self._target_paths = paths
        self._target_pool = ThreadPoolExecutor(max_workers=self.mmd_target_workers)
        self._prefetch_pool = ThreadPoolExecutor(max_workers=1)

    def _load_one_target(self, rel_path):
        img = Image.open(osp.join(self.mmd_target_root, rel_path)).convert('RGB')
        # identical preprocessing to the ImageNet dataset, random_flip included, so
        # the target marginal matches the distribution the FM loss trains on
        # uint8 HWC: the float conversion and the CHW permute happen on the GPU at
        # consumption, so in-flight prefetch buffers and the host->device copy are a
        # quarter the size they would be as float CHW.
        return torch.from_numpy(
            image_preproc(img, self.denoising.input_size, random_flip=True))

    def _pick_target_paths(self, labels, per_row):
        """``per_row`` relpaths per label, drawn uniformly from that label's ENTIRE
        class pool. Pure CPU/python -- safe to run off the main thread."""
        rel = []
        for c in labels:
            c = int(c)
            pool = self._target_paths[c] if 0 <= c < len(self._target_paths) \
                and self._target_paths[c] else None
            if pool is None:  # null label (or an empty class) -> draw from everything
                for _ in range(per_row):
                    cc = random.randrange(len(self._target_paths))
                    while not self._target_paths[cc]:
                        cc = random.randrange(len(self._target_paths))
                    rel.append(random.choice(self._target_paths[cc]))
            else:
                rel.extend(random.choice(pool) for _ in range(per_row))
        return rel

    def _load_target_images(self, labels, per_row):
        """One timestep's worth of target images as a uint8 CPU tensor. Touches no
        CUDA, so it can run entirely on a prefetch thread while the GPU is busy."""
        rel = self._pick_target_paths(labels, per_row)
        if self._target_u8 is not None:
            return self._target_u8.get(rel, flip=np.random.rand(len(rel)) < 0.5)
        return torch.stack(list(self._target_pool.map(self._load_one_target, rel)))

    def _draw_restricted_labels(self, m, device):
        """``m`` trajectory labels spread over ``mmd_classes_per_batch`` DISTINCT
        classes, drawn uniformly without replacement (as d-flow's draw_classes does),
        so each class carries ~m/C trajectories rather than one."""
        c = max(1, min(int(self.mmd_classes_per_batch), m))
        cls = torch.randperm(int(self.denoising.num_classes), device=device)[:c]
        return cls[torch.arange(m, device=device) % c]

    def _prefetch_targets(self, labels, per_row, n_draws):
        """Issue every band timestep's image load UP FRONT, before the flow-matching
        forward/backward runs.

        The draws depend only on the labels -- known here -- and on nothing the
        rollout produces, so they can all be in flight while the GPU works through
        the FM loss and then the band rollout. That is several seconds of compute
        against ~0.5 s of loading per draw, so the reads finish long before the
        scoring loop asks for them and the disk cost disappears from the step time
        rather than merely shrinking.

        Submitted to a single orchestrator thread so the draws run one after another,
        each still fanning out across ``_target_pool``; that keeps the loader from
        oversubscribing the CPU against the dataloader workers already running.
        """
        self._drop_pending()
        self._init_target_index()
        labels = [int(c) for c in labels]     # resolve off the GPU once, not per draw
        for _ in range(n_draws):
            self._pending.append(
                self._prefetch_pool.submit(self._load_target_images, labels, per_row))

    def _drop_pending(self):
        """Discard any unconsumed draws (a claimed iteration that then bailed out)."""
        while self._pending:
            f = self._pending.popleft()
            f.cancel()

    def _n_target_draws(self, x_0, t_split):
        """How many band states will be SCORED, i.e. how many draws to prefetch.
        Mirrors the step accounting in :meth:`_band_rollout` (steps above
        ``mmd_t_hi`` are integrated without gradient and are not scored)."""
        grid = self._build_mmd_sampler(x_0.shape[2:].numel(), x_0.device).sigmas
        n_pre = 0
        if self.mmd_t_hi is not None:
            while 2 * (n_pre + 1) < len(grid) \
                    and float(grid[2 * (n_pre + 1)]) > self.mmd_t_hi:
                n_pre += 1
        n_band = n_pre
        while 2 * (n_band + 1) < len(grid) \
                and float(grid[2 * (n_band + 1)]) >= t_split:
            n_band += 1
        n = max(0, n_band - n_pre)
        # 'bank'/'both' reuse one image draw across the whole band, so prefetching
        # one per step would queue draws the scoring loop never consumes and
        # _drop_pending would cancel them at the end of every iteration.
        return min(n, 1) if self.mmd_target_share != 'none' else n

    @torch.no_grad()
    def _draw_target_latents(self, labels, per_row, device):
        """``per_row`` FRESH images per label, encoded into the diffusion input space.

        Consumes a prefetched draw when one is queued, otherwise loads synchronously
        (first iteration, or if prefetch was skipped). Either way the images are fresh
        for THIS timestep -- the queue holds one independent draw per band state.
        """
        self._init_target_index()
        assert self._encode_fn is not None, (
            'the MMD target side reads real images from disk and needs an encode_fn '
            '(images -> diffusion input space); the wrapper supplies it as '
            'expert_encode_fn. Got None.')
        if self._pending:
            imgs = self._pending.popleft().result()
        else:
            imgs = self._load_target_images([int(c) for c in labels], per_row)
        imgs = imgs.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
        return self._encode_fn(imgs)

    def _target_feats(self, x_on, space, sigma, n_local, labels, shared=None):
        """Features of the target (noised on-path) side.

        With ``mmd_target_n`` set, the images are drawn FRESH FROM DISK for this
        timestep (see :meth:`_draw_target_latents`) and noised with their OWN
        independent draw -- unrelated to the rollout's initial noise, and unrelated to
        the noise any other timestep used. Without it, falls back to the micro-batch
        rows the rollout was built from.
        """
        if n_local is None or space != self.mmd_feature:
            return self._mmd_feats(x_on, space)
        with torch.no_grad():
            if shared is None:              # mmd_target_share='none'
                per_row = max(1, n_local // max(len(labels), 1))
                x0 = self._draw_target_latents(labels, per_row, x_on.device)
                eps = torch.randn_like(x0)  # independent noise, drawn per timestep
            else:                           # 'bank' / 'both': hoisted in _mmd_loss
                x0, eps = shared
                if eps is None:             # 'bank' -> images held, noise per step
                    eps = torch.randn_like(x0)
            x = x0 * (1.0 - sigma) + eps * sigma
            return self._mmd_feats(x, space)

    def _mmd_feats(self, x, space):
        if space == 'subspace':
            return self.feat_fn(x).flatten(1)
        return x.flatten(1)

    def _mmd_loss(self, x_0, class_labels, t_split):
        """Per-NFE-step MMD^2 between the rollout marginal and the noised on-path
        marginal, averaged over the band's steps."""
        m = min(self.mmd_batch, x_0.size(0))
        x_0 = x_0[:m].detach()
        labels = class_labels[:m]

        # NB eval mode / compile bypass are applied per-eval inside _ckpt_pred, not
        # around this call: they have to be in force again when the checkpoints
        # recompute during backward, which happens after this method has returned.
        states = self._band_rollout(x_0, labels, t_split)

        train_spaces = ('subspace', 'raw') if self.mmd_feature == 'both' \
            else (self.mmd_feature, )
        per_space = dict(subspace=[], raw=[])
        loss = x_0.new_zeros(())
        world_size, n_pooled = 1, m  # overwritten per step once the gather is known

        # keep a FIFO of this rank's real-latent FEATURES so the target side can draw
        # on more distinct images than one micro-batch holds (see mmd_target_n).
        # Updated once per call, before the states are scored. Applies to BOTH
        # spaces: 'raw' caches bf16 latents, 'subspace' the 2048-d features.
        n_local = None
        if self.mmd_target_n is not None:
            ws = dist.get_world_size() if (
                self.mmd_gather and dist.is_available() and dist.is_initialized()) else 1
            n_local = max(1, int(self.mmd_target_n) // ws)

        # 'bank' / 'both': ONE draw for the whole band, taken before the loop so the
        # reused parts are identical at every step by construction rather than by the
        # draw order happening to line up.
        shared = None
        if self.mmd_target_share != 'none' and n_local is not None:
            with torch.no_grad():
                x1_s = self._draw_target_latents(
                    [int(c) for c in labels],
                    max(1, n_local // max(len(labels), 1)), x_0.device)
            shared = (x1_s,
                      torch.randn_like(x1_s) if self.mmd_target_share == 'both' else None)

        for idx, (sigma, x_roll) in enumerate(states):
            # forward-diffusion marginal at the SAME sigma, from the same rows
            # (identical class mixture), fresh noise per step
            x_on = x_0 * (1.0 - sigma) + torch.randn_like(x_0) * sigma
            for space in ('subspace', 'raw'):
                if space in train_spaces:
                    # gather AFTER the feature map: for the subspace that is 2048-d
                    # per sample instead of the 196608-d state, and the local rows
                    # keep their graph either way.
                    f_roll, world_size = self._gather_feats(self._mmd_feats(x_roll, space))
                    f_on, _ = self._gather_feats(self._target_feats(x_on, space, sigma, n_local, labels, shared))
                    val = mmd2_rbf(
                        f_roll, f_on,
                        bandwidths=self.mmd_bandwidths, unbiased=self.mmd_unbiased)
                    loss = loss + val
                else:  # not trained on -- logged only
                    with torch.no_grad():
                        f_roll, world_size = self._gather_feats(self._mmd_feats(x_roll, space))
                        f_on, _ = self._gather_feats(self._target_feats(x_on, space, sigma, n_local, labels, shared))
                        val = mmd2_rbf(
                            f_roll, f_on,
                            bandwidths=self.mmd_bandwidths, unbiased=self.mmd_unbiased)
                n_pooled = f_roll.shape[0]
                per_space[space].append(val.detach())

        # /n_steps -> mean over the band's steps. Two correction factors on top,
        # both so the applied gradient is exactly d(mean_k MMD^2_k)/d(theta) whatever
        # the parallel layout:
        #   *mmd_accum_steps  cancels train_grad_accum's 1/N (this term runs on one
        #                     micro-batch, not all N)
        #   *world_size       cancels DDP's gradient averaging: with a pooled MMD each
        #                     rank holds the same loss but differentiates only through
        #                     its own rows, so the per-rank gradients SUM to the true
        #                     gradient and averaging them would shrink it by 1/W.
        n_steps = max(len(states), 1)
        loss = loss * (self.mmd_accum_steps * world_size / n_steps)

        log_vars = dict()
        for space, vals in per_space.items():
            if not vals:
                continue
            tag = 'mmd_sub' if space == 'subspace' else 'mmd_raw'
            acc = float(self.mmd_accum_steps)  # see `scale` above: read as true MMD^2
            log_vars[tag] = torch.stack(vals).mean() * acc
            if space in train_spaces:  # per-step detail for the trained space only
                for idx, (sigma, _) in enumerate(states):
                    log_vars[f'{tag}_s{sigma:.3f}'] = vals[idx] * acc
        log_vars['mmd_classes'] = loss.new_tensor(
            float(len(set(labels.tolist()))) * acc)
        log_vars['mmd_steps'] = loss.new_tensor(
            float(len(states)) * self.mmd_accum_steps)
        # trajectories actually entering each MMD estimate (world_size * mmd_batch
        # when gathering) -- the estimator's noise floor scales like 1/this.
        log_vars['mmd_n'] = loss.new_tensor(float(n_pooled) * self.mmd_accum_steps)
        return loss, log_vars

    def _mmd_due(self, running_status):
        """Whether this ITERATION should carry the MMD term. Pure predicate -- no
        side effects, so it is safe to ask more than once per step."""
        if self.mmd_weight == 0:
            return False
        if running_status is None:
            return True
        it = running_status.get('iteration', 0)
        return it >= self.mmd_start_iter and it % self.mmd_interval == 0

    def _mmd_claim(self, running_status):
        """True at most ONCE per iteration: the band runs on a single micro-batch,
        so the first caller claims it and later ones are turned away.

        Kept separate from _mmd_due deliberately. When the two were one method, the
        warmup cache-fill check asked 'is MMD active?' first, BURNED the latch, and
        the real gate then saw the iteration as already claimed -- so the MMD loss was
        silently skipped on every iteration it was meant to run. Below mmd_start_iter
        the predicate returned False without latching, so the warmup path still
        worked and the bug was invisible in isolation."""
        if not self._mmd_due(running_status):
            return False
        if running_status is None:
            return True
        it = running_status.get('iteration', 0)
        if self._mmd_done_iter == it:
            return False
        self._mmd_done_iter = it
        return True

    def _grad_probe(self, loss_fm, loss_mmd):
        """Parameter-gradient norms of the two terms, measured separately.
        ``gnorm_mmd`` is for the UNWEIGHTED MMD, so ``mmd_weight`` scaled to
        ``gnorm_fm / gnorm_mmd`` would make the two terms push equally hard."""
        params = [p for p in self.denoising.parameters() if p.requires_grad]

        def gnorm(x):
            grads = torch.autograd.grad(
                x, params, retain_graph=True, allow_unused=True)
            sq = [g.detach().float().pow(2).sum() for g in grads if g is not None]
            return torch.stack(sq).sum().sqrt() if sq else x.new_zeros(())

        g_fm, g_mmd = gnorm(loss_fm), gnorm(loss_mmd)
        return dict(
            gnorm_fm=g_fm,
            gnorm_mmd=g_mmd,
            gnorm_ratio=g_fm / g_mmd.clamp_min(1e-12))

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
        if expert_encode_fn is not None:
            self._encode_fn = expert_encode_fn

        # Decide + prefetch BEFORE the flow-matching step, not after: the claim is
        # what makes this the micro-batch that runs MMD, and issuing the draws here
        # buys them the whole FM forward/backward plus the band rollout to complete.
        t_split = self.mmd_t_split if self.mmd_t_split is not None else self.t_split
        run_mmd = (self._mmd_claim(running_status) and t_split is not None
                   and 'class_labels' in kwargs)
        mmd_labels = None
        if run_mmd:
            mmd_labels = class_labels_true if class_labels_true is not None \
                else kwargs['class_labels']
            if self.mmd_classes_per_batch is not None:
                # drawn ONCE here and reused below: a second draw would prefetch one
                # class set and then score a different one
                mmd_labels = self._draw_restricted_labels(
                    min(self.mmd_batch, x_0.size(0)), x_0.device)
        if run_mmd and self.mmd_target_n is not None:
            m = min(self.mmd_batch, x_0.size(0))
            ws = dist.get_world_size() if (
                self.mmd_gather and dist.is_available() and dist.is_initialized()) else 1
            n_local = max(1, int(self.mmd_target_n) // ws)
            self._prefetch_targets(mmd_labels[:m], max(1, n_local // max(m, 1)),
                                   self._n_target_draws(x_0, t_split))

        loss, log_vars = super().forward_train(
            x_0,
            visual_encoder_features=visual_encoder_features,
            running_status=running_status,
            buffer_batch=buffer_batch,
            n_onpath_high=n_onpath_high,
            **kwargs)

        # purely online by construction: the band rollout is generated fresh from new
        # noise every iteration and discarded. No replay buffer is involved -- the
        # inherited DAGGER buffer stream must stay inert for this arm.
        assert buffer_batch is None or self.mmd_weight == 0, (
            'GaussianFlowMMD is an online objective: the MMD is taken against the '
            'policy\'s CURRENT marginals, so a replay buffer of stale rollout states '
            'has no place in it. Got a buffer_batch -- drop the DaggerRolloutHook '
            '(and set expert=None).')
        if run_mmd:
            # CONDITIONAL-ONLY rollouts: use the UNDROPPED labels. kwargs['class_labels']
            # has had CFG dropout applied, so ~10% of trajectories would be generated
            # under the null label and the band would mix conditional samples with
            # unconditional ones. The wrapper supplies the true labels (it inspects
            # this signature for class_labels_true); falling back to the dropped ones
            # only if it did not. Matches the CFG arms, which also roll out
            # conditional-only and apply dropout inside the loss instead.
            mmd_loss, mmd_log_vars = self._mmd_loss(x_0, mmd_labels, t_split)
            self._drop_pending()   # nothing should remain, but never leak into the next iter
            if self.mmd_grad_probe:
                mmd_log_vars.update(self._grad_probe(loss, mmd_loss))
            loss = loss + self.mmd_weight * mmd_loss
            log_vars.update(mmd_log_vars)
            log_vars['loss_mmd'] = (self.mmd_weight * mmd_loss).detach()

        return loss, log_vars
