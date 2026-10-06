# Copyright (c) 2026 Hansheng Chen

import contextlib
import os.path as osp
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from lakonlab.datasets.imagenet import image_preproc
from .gaussian_flow_mmd import mmd2_energy, mmd2_rbf, rbf_width_stats

_U8_DEFAULT = '/dev/shm/asymflow/train_u8_256'
_INDEX_CACHE = dict()   # datalist path -> per-class relpath lists (shared by instances)


@contextlib.contextmanager
def _no_tf32():
    """Full fp32 matmuls inside. tools/test.py turns TF32 on globally, and the raw
    pairwise distances are ~2e5-magnitude dot products whose TF32 rounding error is the
    same order as the differences the gradient is made of."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def _dist_on():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def mmd2_point_rbf(x, y, bandwidths=(0.25, 0.5, 1.0, 2.0, 4.0), eps=1e-12, width='mean',
                   normalize=False):
    """Per-row MMD^2 between the point mass at ``x[b]`` and the K-sample set ``y[b]``:
    ``x [B, D]``, ``y [B, K, D]`` -> ``[B]``.

        MMD^2(delta_x, Q_b) = k(x, x) - 2 mean_k k(x, y_bk) + mean_{k!=j} k(y_bk, y_bj)

    with :func:`mmd2_rbf`'s kernel: the same multiplier mixture on a base width equal to
    the mean squared distance between two DISTINCT target samples -- here measured per
    row, on that row's own target set, so the width is the target distribution's and
    nothing else's. k(x, x) = 1 and the target-target term are constant in x, so the
    gradient is the kernel-weighted pull toward the targets alone; they are kept so the
    logged value is the actual MMD^2 (>= 0 up to the unbiased target term's noise).

    ``width='spread'``: :func:`mmd2_rbf`'s spread widths, from each row's own target
    distances. There k(x, x) = exp(mean / (mult * sd)) is astronomically large (and
    constant in x), so it is left out, and each multiplier's term is divided by the
    row's own target-target mean E_yy[k] (as :func:`mmd2_rbf` does): the value is
    1 - 2 E[k(x, y)] / E_yy[k] per width (can be negative). Needs K >= 4 targets per
    row: with 2 the single target pair's z-score is degenerate (sd of one value).
    ``normalize=True`` does the same offset / self-term drop / per-multiplier ratio with
    the 'mean' widths (:func:`mmd2_rbf`'s ``normalize``).
    """
    x = x.float()
    y = y.float()
    K = y.shape[1]
    assert K > 1, f'per-sample MMD^2 needs at least 2 targets per row, got {K}.'
    assert width == 'mean' or K >= 4, (
        f"width='spread' needs >= 4 targets per row (got {K}): the per-row d2 sd is "
        'degenerate with fewer.')
    with torch.no_grad():
        yy = torch.bmm(y, y.transpose(1, 2))                           # [B, K, K]
        y2 = yy.diagonal(dim1=1, dim2=2)                                # [B, K]
        d_yy = (y2.unsqueeze(2) + y2.unsqueeze(1) - 2.0 * yy).clamp_min(0)
        off = ~torch.eye(K, dtype=torch.bool, device=y.device)
        stable = width == 'spread' or normalize
        offset, scale = rbf_width_stats(d_yy, width, eps, offset_mean=stable)  # [B], [B]
    xy = torch.bmm(y, x.unsqueeze(2)).squeeze(2)                        # [B, K]
    d_xy = (x.pow(2).sum(1, keepdim=True) + y2 - 2.0 * xy).clamp_min(0)
    self_k = 0.0 if stable else 1.0
    total = 0.0
    for mult in bandwidths:
        denom = (scale * mult).unsqueeze(1)                             # [B, 1]
        with torch.no_grad():
            e_yy = torch.exp(-((d_yy - offset.view(-1, 1, 1)) / denom.unsqueeze(2)
                               ).clamp_min(-60.0))[:, off].mean(1)
        k_xy = torch.exp(-((d_xy - offset.unsqueeze(1)) / denom).clamp_min(-60.0))
        if stable:   # per-multiplier normalisation, as mmd2_rbf
            total = total + 1.0 - 2.0 * k_xy.mean(1) / e_yy.clamp_min(eps)
        else:
            total = total + self_k + e_yy - 2.0 * k_xy.mean(1)
    return total / len(bandwidths)


def mmd2_point_energy(x, y, eps=1e-12, beta=1.0):
    """Per-row energy-distance MMD^2 (:func:`mmd2_energy`'s kernel -||a - b||) between
    the point mass at ``x[b]`` and the K-sample set ``y[b]``: ``x [B, D]``,
    ``y [B, K, D]`` -> ``[B]``.

        E(delta_x, Q_b) = 2 mean_k ||x - y_bk|| - mean_{k!=j} ||y_bk - y_bj||

    (||x - x|| = 0). The target term is constant in x; it is kept so the value is the
    actual statistic. No bandwidth; ``beta`` is the distance exponent (see
    :func:`mmd2_energy`).
    """
    assert 0.0 < beta < 2.0, f'energy beta must be in (0, 2), got {beta}'
    x = x.float()
    y = y.float()
    K = y.shape[1]
    assert K > 1, f'per-sample MMD^2 needs at least 2 targets per row, got {K}.'
    with torch.no_grad():
        yy = torch.bmm(y, y.transpose(1, 2))                           # [B, K, K]
        y2 = yy.diagonal(dim1=1, dim2=2)                                # [B, K]
        d_yy = (y2.unsqueeze(2) + y2.unsqueeze(1) - 2.0 * yy).clamp_min(0)
        off = ~torch.eye(K, dtype=torch.bool, device=y.device)
        e_yy = d_yy[:, off].clamp_min(eps).pow(0.5 * beta).mean(1)     # [B]
    xy = torch.bmm(y, x.unsqueeze(2)).squeeze(2)                        # [B, K]
    d_xy = (x.pow(2).sum(1, keepdim=True) + y2 - 2.0 * xy).clamp_min(eps)
    return 2.0 * d_xy.pow(0.5 * beta).mean(1) - e_yy


class MMDGuidance:
    """Inference-time MMD guidance: at every full solver step, before the network eval,
    each rollout sample takes one normalized gradient step down an MMD^2 against the
    TARGET distribution -- real images of the sample's own class, noised to the same
    sigma, ``(1 - sigma) x_0 + sigma eps`` -- in raw flattened latents, with the
    training term's kernel (:func:`mmd2_rbf`'s bandwidth rule). The step is

        g      = d MMD^2 / d x
        x'     = x - scale * g / ||g||_2 * ||x||_2
        x     <- x' / ||x'||_2 * ||x||_2                       (``renorm``)

    Two objectives:

      ``'sample'`` (default) -- each sample is scored ALONE, as the point mass at x_i
          against its own ``target_per_row`` class-matched targets
          (:func:`mmd2_point_rbf`). Its perturbation depends only on the target
          distribution, never on the other rollouts or on how many there are: no
          cross-sample repulsion term and no cross-rank communication. Norms are
          per sample by default.
      ``'pooled'`` -- one MMD between the whole rollout batch and the whole target set,
          pooled over every DDP rank (the first version). The gradient for x_i then
          includes a repulsion from the OTHER rollouts, a finite-sample term whose
          size is set by the rollout count; at 512 rollouts vs 2048 targets the pooled
          MMD^2 sat at its noise floor (negative on average).

    ``scale`` is the step's displacement as a fraction of the state's norm, independent
    of how large the raw gradient is. The renormalization puts the state back on its
    pre-step norm: at high sigma ||x|| is essentially the noise level (sqrt(D) *
    sigma), so a step with a radial component would otherwise hand the network a state
    at the wrong noise level, and the drift compounds over 50 steps. The norm drift it
    would have caused is logged (``mmdg_norm_ratio``). NB the step has the same size
    whatever the gradient's magnitude -- at sigma = 1 the targets are pure noise and the
    direction carries no data information, but it is still applied at full size.
    ``sigma_range`` excludes such steps.

    Driven by ``GaussianFlow.forward_test(state_callback=...)``, which calls it at
    every FIRST-ORDER solver state: for Heun, each landing state sigma_k before its
    predictor eval (50 calls at 50 NFE-steps, sigma = 1.00 .. 0.02), never between
    predictor and corrector. Under ``objective='pooled'`` or ``norm='batch'`` every
    rank must reach every call in lockstep -- the eval DistributedSampler pads to equal
    full batches, so it does.

    Args:
        scale (float): criteria_guidance_scale, the per-step displacement as a
            fraction of ||x||. 0 makes the call an identity.
        objective (str): ``'sample'`` or ``'pooled'``, see above.
        sigma_range (tuple): ``[lo, hi]``; only steps with lo <= sigma <= hi are
            guided (and only those draw targets).
        norm (str | None): ``'sample'`` (per row) or ``'batch'`` (pooled over every
            rank). Applies to both the step size and ``renorm``. None -> ``'sample'``
            for the per-sample objective, ``'batch'`` for the pooled one.
        renorm (bool): rescale x after the step back to its pre-step norm, per row or
            pooled as ``norm`` says.
        target_per_row (int | None): real images drawn per rollout sample per step --
            for ``'sample'``, the size of each sample's own target set. None -> 16 for
            ``'sample'``, 4 for ``'pooled'``.
        kernel (str): ``'rbf'`` (the multi-bandwidth Gaussian mixture, default) or
            ``'energy'`` (the distance kernel -||a - b||: energy distance, no
            bandwidth -- :func:`mmd2_energy` / :func:`mmd2_point_energy`).
        energy_beta (float): the energy kernel's distance exponent, in (0, 2).
        feature (str): the space the MMD is computed in -- ``'raw'`` (flattened
            diffusion-space latents, 196,608-d) or ``'subspace'`` (the diffusion's
            ``feat_fn``: AsymJiT's rank-8 per-patch features, 2048-d, as
            GaussianFlowMMD's ``mmd_feature='subspace'``). The step, its norm and the
            renorm stay on the full state; feat_fn is linear, so the gradient is the
            subspace-feature gradient mapped back through the projection.
        normalize (bool): divide each width's term by its target-target mean
            (:func:`mmd2_rbf`), so a mixture of very different widths is balanced.
        width (str): kernel widths from the targets' pairwise squared distances --
            ``'mean'`` (multiples of their mean, the original rule) or ``'spread'``
            (multiples of their standard deviation; see :func:`mmd2_rbf`).
        target_share (str): ``'none'`` fresh images and noise every step; ``'bank'``
            one image draw per batch reused at every step, noise per step; ``'both'``
            images and noise fixed for the batch (targets are straight-line
            trajectories). As GaussianFlowMMD's ``mmd_target_share``.
        u8_cache (str | None): preprocessed uint8 image cache prefix; ``'auto'`` uses
            the node-local /dev/shm cache when it is complete, else JPEGs.
    """

    def __init__(self,
                 scale,
                 objective='sample',
                 sigma_range=(0.0, 1.0),
                 norm=None,
                 renorm=True,
                 target_per_row=None,
                 target_share='none',
                 bandwidths=(0.25, 0.5, 1.0, 2.0, 4.0),
                 width='mean',
                 normalize=False,
                 feature='raw',
                 kernel='rbf',
                 energy_beta=1.0,
                 unbiased=True,
                 gather=True,
                 datalist='data/imagenet/train.txt',
                 data_root='data/imagenet/train/',
                 u8_cache='auto',
                 image_size=256,
                 num_classes=1000,
                 load_workers=32,
                 read_threads=8,
                 lookahead=2,
                 log_sigmas=(0.98, 0.88, 0.5, 0.1, 0.02),
                 measure_only=False,
                 seed=0):
        assert objective in ('sample', 'pooled'), objective
        if norm is None:
            norm = 'sample' if objective == 'sample' else 'batch'
        if target_per_row is None:
            target_per_row = 16 if objective == 'sample' else 4
        assert norm in ('batch', 'sample'), norm
        assert target_share in ('none', 'bank', 'both'), target_share
        self.scale = float(scale)
        # measure_only: score the UNGUIDED state at every call in sigma_range, never
        # step (scale is ignored), and log the MMD^2 and its square at EVERY grid sigma,
        # so the eval's batch average gives the per-sigma mean and its spread
        self.measure_only = bool(measure_only)
        self.objective = objective
        self.sigma_range = tuple(float(v) for v in sigma_range)
        self.norm = norm
        self.renorm = bool(renorm)
        self.target_per_row = int(target_per_row)
        self.target_share = target_share
        self.bandwidths = tuple(bandwidths)
        assert width in ('mean', 'spread'), width
        self.width = width
        assert feature in ('raw', 'subspace'), feature
        self.feature = feature
        assert width == 'mean' or objective == 'pooled' or int(target_per_row or 16) >= 4, (
            "width='spread' with objective='sample' needs target_per_row >= 4.")
        assert width == 'mean' or unbiased, "width='spread' needs the unbiased estimator."
        assert not normalize or unbiased, 'normalize needs the unbiased estimator.'
        self.normalize = bool(normalize)
        assert kernel in ('rbf', 'energy'), kernel
        assert kernel == 'rbf' or width == 'mean', (
            "the energy kernel has no width; leave width at its default.")
        self.kernel = kernel
        assert 0.0 < float(energy_beta) < 2.0, energy_beta
        self.energy_beta = float(energy_beta)
        self.unbiased = unbiased
        self.gather = gather
        self.datalist = datalist
        self.data_root = data_root
        if u8_cache == 'auto':
            u8_cache = _U8_DEFAULT if osp.exists(_U8_DEFAULT + '.complete') else None
        from lakonlab.datasets.u8_cache import U8ImageCache
        self._u8 = U8ImageCache(u8_cache) if u8_cache else None
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.load_workers = int(load_workers)
        self.read_threads = int(read_threads)
        self.lookahead = max(1, int(lookahead))
        self.log_sigmas = tuple(log_sigmas)
        rank = dist.get_rank() if _dist_on() else 0
        # own generators: target draws run on a prefetch thread, so they must not
        # share (or perturb) the global RNG streams the sampler's noise comes from
        self._rng = random.Random(seed * 1000003 + rank)
        self._np_rng = np.random.default_rng(seed * 1000003 + rank)
        self._paths = None
        self._load_pool = None
        self._prefetch_pool = None
        self._pending = deque()
        self._labels = None
        self._encode_fn = None
        self._shared = None
        self._stats = None

    # ---- target images ------------------------------------------------------

    def _init_index(self):
        if self._paths is not None:
            return
        if self.datalist not in _INDEX_CACHE:
            paths = [[] for _ in range(self.num_classes)]
            with open(self.datalist) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2 and 0 <= int(parts[1]) < self.num_classes:
                        paths[int(parts[1])].append(parts[0])
            _INDEX_CACHE[self.datalist] = paths
        self._paths = _INDEX_CACHE[self.datalist]
        if self._u8 is None:
            self._load_pool = ThreadPoolExecutor(max_workers=self.load_workers)
        self._prefetch_pool = ThreadPoolExecutor(max_workers=1)

    def _pick(self, labels):
        """``target_per_row`` relpaths per label from that class's whole pool, row-major
        (row b's targets are entries ``b*K .. b*K+K-1``); a null (or out-of-range) label
        draws from the whole dataset, class first."""
        rel = []
        for c in labels:
            for _ in range(self.target_per_row):
                cc = c
                while not (0 <= cc < self.num_classes and self._paths[cc]):
                    cc = self._rng.randrange(self.num_classes)
                rel.append(self._rng.choice(self._paths[cc]))
        return rel

    def _load_one(self, rel_path):
        img = Image.open(osp.join(self.data_root, rel_path)).convert('RGB')
        return torch.from_numpy(image_preproc(img, self.image_size, random_flip=True))

    def _load(self, labels):
        """uint8 ``[M, H, W, 3]`` CPU tensor, flipped with p=0.5 like the train set."""
        rel = self._pick(labels)
        if self._u8 is not None:
            # pread straight into one buffer on read_threads threads -- the per-sample
            # objective reads K=16 images per row per step, ~200 MB per rank per step
            out = np.empty((len(rel), ) + self._u8.image_shape, dtype=np.uint8)
            self._u8.read_into(rel, out, threads=self.read_threads)
            flip = self._np_rng.random(len(rel)) < 0.5
            if flip.any():
                out[flip] = out[flip][:, :, ::-1]
            return torch.from_numpy(out)
        return torch.stack(list(self._load_pool.map(self._load_one, rel)))

    def _submit(self):
        self._pending.append(self._prefetch_pool.submit(self._load, self._labels))

    def _drop_pending(self):
        while self._pending:
            self._pending.popleft().cancel()

    def _draw_x0(self, device):
        imgs = self._pending.popleft().result()
        if self.target_share == 'none':
            self._submit()     # keep the lookahead full
        imgs = imgs.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
        return self._encode_fn(imgs)

    # ---- batch lifecycle ----------------------------------------------------

    def begin(self, labels, encode_fn):
        """Start a new batch: ``labels`` are the rollout rows' labels (the null label
        for unconditional rows), ``encode_fn`` maps [0, 1] images to the diffusion
        input space."""
        self._init_index()
        self._drop_pending()
        self._labels = [int(c) for c in labels.tolist()]
        self._encode_fn = encode_fn
        self._shared = None
        self._stats = dict(before=[], after=[], sigma=[], norm=[])
        n = 1 if self.target_share != 'none' else self.lookahead
        for _ in range(n):
            self._submit()

    def end(self):
        """Per-batch diagnostics as floats (``evaluate`` averages them over batches and
        ranks): the MMD^2 before and after each guided step -- the mean over samples
        of the per-sample MMD^2 for ``'sample'``, the pooled MMD^2 for ``'pooled'`` --
        mean over steps and at the grid states nearest ``log_sigmas``. 'after'
        re-scores the step against the SAME target draw its gradient came from, so it
        checks that the step descends; it is not an independent estimate.
        ``mmdg_norm_ratio`` is ||x'|| / ||x|| BEFORE renormalization (logged either
        way; 1 = the step was purely tangential and renorm had nothing to undo)."""
        self._drop_pending()
        st = self._stats
        out = dict()
        if not st or not st['sigma']:
            return out
        out['mmdg_mmd'] = float(np.mean(st['before']))
        out['mmdg_mmd_after'] = float(np.mean(st['after']))
        out['mmdg_norm_ratio'] = float(np.mean(st['norm']))
        # cumulative drift renorm prevented (or, with renorm off, the drift applied)
        out['mmdg_norm_ratio_prod'] = float(np.prod(st['norm']))
        sig = np.asarray(st['sigma'])
        if self.measure_only:
            for s, v in zip(sig, st['before']):
                out[f'mmdg_mmd_s{s:.2f}'] = float(v)
                out[f'mmdg_mmdsq_s{s:.2f}'] = float(v) ** 2
            return out
        for s in self.log_sigmas:
            k = int(np.abs(sig - s).argmin())
            if abs(sig[k] - s) < 0.011:
                out[f'mmdg_mmd_s{sig[k]:.2f}'] = float(st['before'][k])
                out[f'mmdg_mmd_after_s{sig[k]:.2f}'] = float(st['after'][k])
        return out

    # ---- the step -----------------------------------------------------------

    def _gather(self, f):
        """All-gather along dim 0; the LOCAL slot keeps ``f``'s graph, the other ranks'
        rows enter as constants (as GaussianFlowMMD._gather_feats)."""
        if not self.gather or not _dist_on():
            return f
        buf = [torch.empty_like(f) for _ in range(dist.get_world_size())]
        dist.all_gather(buf, f.detach().contiguous())
        buf[dist.get_rank()] = f
        return torch.cat(buf, dim=0)

    def _all_sum(self, v):
        if self.gather and _dist_on():
            dist.all_reduce(v)
        return v

    def _feat(self, diffusion, z):
        """``[M, C, H, W]`` states -> ``[M, F]`` MMD features (see ``feature``)."""
        if self.feature == 'subspace':
            assert hasattr(diffusion, 'feat_fn'), (
                "feature='subspace' needs a diffusion with feat_fn (GaussianFlowDagger).")
            return diffusion.feat_fn(z).flatten(1)
        return z.flatten(1)

    def _score(self, diffusion, x, y):
        """The objective's MMD^2 for the rollout state ``x [B, ...]`` against the noised
        target FEATURES ``y`` (``[B, K, F]`` for 'sample', pooled ``[N, F]`` for
        'pooled'). Returns ``(value to differentiate, value to log)``."""
        fx = self._feat(diffusion, x)
        if self.objective == 'sample':
            per_row = mmd2_point_energy(fx, y, beta=self.energy_beta) if self.kernel == 'energy' else \
                mmd2_point_rbf(fx, y, bandwidths=self.bandwidths, width=self.width,
                               normalize=self.normalize)
            # rows are independent, so the gradient of the sum is each row's own
            return per_row.sum(), float(per_row.detach().mean())
        if self.kernel == 'energy':
            val = mmd2_energy(self._gather(fx), y, unbiased=self.unbiased, beta=self.energy_beta)
        else:
            val = mmd2_rbf(self._gather(fx), y,
                           bandwidths=self.bandwidths, unbiased=self.unbiased,
                           width=self.width, normalize=self.normalize)
        return val, float(val.detach())

    def __call__(self, diffusion, x_t, t):
        sigma = float(t) / diffusion.num_timesteps
        lo, hi = self.sigma_range
        if (self.scale == 0 and not self.measure_only) or not (lo - 1e-6 <= sigma <= hi + 1e-6):
            return x_t
        device = x_t.device
        with torch.no_grad():
            if self.target_share == 'none':
                x0, eps = self._draw_x0(device), None
            else:
                if self._shared is None:
                    x0 = self._draw_x0(device)
                    self._shared = (x0, torch.randn_like(x0)
                                    if self.target_share == 'both' else None)
                x0, eps = self._shared
            if eps is None:
                eps = torch.randn_like(x0)
            y = self._feat(diffusion, (x0 * (1.0 - sigma) + eps * sigma).float()).float()
            if self.objective == 'sample':
                y = y.view(x_t.shape[0], self.target_per_row, -1)   # row b's own targets
            else:
                y = self._gather(y)

        if self.measure_only:
            with torch.no_grad(), _no_tf32():
                _, before = self._score(diffusion, x_t.detach().float(), y)
            for k, v in (('sigma', sigma), ('before', before), ('after', before), ('norm', 1.0)):
                self._stats[k].append(v)
            return x_t

        with torch.enable_grad(), _no_tf32():
            x = x_t.detach().float().requires_grad_(True)
            val, before = self._score(diffusion, x, y)
            g, = torch.autograd.grad(val, x)

        with torch.no_grad(), _no_tf32():
            x = x.detach()
            if self.norm == 'batch':
                sq = self._all_sum(torch.stack([g.pow(2).sum(), x.pow(2).sum()]))
                xn = sq[1].sqrt()
                x_new = x - g * (self.scale * xn / sq[0].sqrt().clamp_min(1e-30))
                # ||x'|| / ||x||, pooled -- the drift renorm removes
                ratio = self._all_sum(x_new.pow(2).sum()).sqrt() / xn.clamp_min(1e-30)
                if self.renorm:
                    x_new = x_new / ratio.clamp_min(1e-30)
                drift = float(ratio)
            else:
                dims = tuple(range(1, x.dim()))
                gn = g.pow(2).sum(dims, keepdim=True).sqrt().clamp_min(1e-30)
                xn = x.pow(2).sum(dims, keepdim=True).sqrt()
                x_new = x - g * (self.scale * xn / gn)
                ratio = x_new.pow(2).sum(dims, keepdim=True).sqrt() / xn.clamp_min(1e-30)
                if self.renorm:
                    x_new = x_new / ratio.clamp_min(1e-30)
                drift = float(ratio.mean())   # local; evaluate averages over ranks
            _, after = self._score(diffusion, x_new, y)
        self._stats['sigma'].append(sigma)
        self._stats['before'].append(before)
        self._stats['after'].append(after)
        self._stats['norm'].append(drift)
        return x_new.to(x_t.dtype)
