# Copyright (c) 2026 Hansheng Chen

import torch
import torch.nn as nn

from ...builder import MODULES


@MODULES.register_module()
class EmpiricalExpert(nn.Module):
    """Per-class posterior-mean data oracle for DAGGER-style flow-matching finetuning.

    Keeps a reservoir of real training latents, organised as one ring buffer
    *per class* (``per_class_pool`` slots each, ``num_classes`` classes), filled
    from the training minibatches so there is no extra data IO. This guarantees a
    balanced per-class pool (every class has up to ``per_class_pool`` members)
    rather than the fluctuating count a single shared reservoir would give.

    There is never a ground-truth ``x_0`` for a visited rollout state -- the
    expert returns the Bayes posterior *mean* over a (class-restricted) bank
    under the codebase convention

        x_t = (1 - sigma) * x_0 + sigma * noise      (x_0 = data, sigma in [0, 1])

    so for any visited state ``x_t`` at level ``sigma``:

        w_i  propto  exp(-|| feat(x_t) - (1 - sigma) * feat(x_0^i) ||^2 / (2 sigma^2))
        x0_hat = sum_i w_i * x_0^i

    Two bank policies:
      - :meth:`sample_bank_idx` + :meth:`x0_hat` -- one sampled ``bank_k`` bank per
        row (used for rollout trajectories, pinned across their timesteps);
      - :meth:`x0_hat_full` -- the ENTIRE class-restricted pool per row (used for
        on-path points; the null/unconditional class is capped at ``bank_k``).
    Both restrict to the row's class (null / unseen class -> whole pool,
    unconditional). ``feat`` is a compact (rank ``basis_rank``) projection from
    the model (``feat_fn``); scoring in that subspace keeps the posterior
    well-conditioned. The full-resolution weighted sum is accumulated in blocks
    of ``bank_chunk`` to bound memory.

    Storage: global index ``= class * per_class_pool + slot``. Full latents live
    on CPU (``num_classes * per_class_pool`` of them -- e.g. ~50 GB for
    1000*64 at 3x256x256), compact features on the compute device. Everything is
    per-process (each DDP rank keeps its own reservoir) and never checkpointed.

    Args:
        num_classes (int): number of classes (reservoir blocks).
        per_class_pool (int): reservoir slots kept per class.
        bank_k (int): candidates per row for the sampled-bank path / the cap for
            the unconditional full path.
        sample_chunk (int): rows processed at once (bounds peak memory).
        bank_chunk (int): bank entries summed at once (bounds peak memory).
        min_sigma (float): floor on ``sigma`` in the posterior denominator.
    """

    def __init__(self,
                 num_classes=1000,
                 per_class_pool=64,
                 bank_k=1024,
                 sample_chunk=32,
                 bank_chunk=128,
                 min_sigma=1e-3):
        super().__init__()
        self.num_classes = num_classes
        self.per_class_pool = per_class_pool
        self.bank_k = bank_k
        self.sample_chunk = sample_chunk
        self.bank_chunk = bank_chunk
        self.min_sigma = min_sigma

        # plain attributes (NOT buffers): keep the big reservoir off the GPU and
        # out of state_dict / .to() sweeps. Flat storage, per-class ring buffers:
        # global index = class * per_class_pool + slot.
        self._pool = None       # [num_classes*per_class_pool, C, H, W] full latents, CPU
        self._pool_feat = None  # [num_classes*per_class_pool, Df] compact features, device
        self._filled = torch.zeros(num_classes, dtype=torch.long)  # per-class valid count
        self._ptr = torch.zeros(num_classes, dtype=torch.long)     # per-class write pointer
        self._all_idx_cache = None  # cached union of filled global indices

    @property
    def ready(self):
        return int(self._filled.sum()) >= max(self.bank_k, 1)

    def _class_indices(self, ci):
        """Global reservoir indices currently filled for class ``ci`` (or None)."""
        fc = int(self._filled[ci])
        if fc == 0:
            return None
        return ci * self.per_class_pool + torch.arange(fc)

    def _all_indices(self):
        """Union of all filled global indices (unconditional fallback), cached."""
        if self._all_idx_cache is None:
            K = self.per_class_pool
            parts = [ci * K + torch.arange(int(self._filled[ci]))
                     for ci in range(self.num_classes) if int(self._filled[ci]) > 0]
            self._all_idx_cache = torch.cat(parts) if parts else torch.arange(0)
        return self._all_idx_cache

    @torch.no_grad()
    def push_pool(self, latents, feat_fn, labels):
        """Add a batch of training latents ``[B, C, H, W]`` (true class ``labels``
        ``[B]``) into their per-class ring buffers.

        ``feat_fn`` maps ``[*, C, H, W] -> [*, Df]`` compact subspace features.
        Uses the true data labels (not CFG-dropout labels) so per-class banks are
        drawn from genuine same-class data.
        """
        latents = latents.detach()
        dev = latents.device
        labels_cpu = labels.detach().to('cpu').long()
        feats = feat_fn(latents).flatten(1).detach()  # [B, Df]
        K = self.per_class_pool
        if self._pool is None:
            self._pool = latents.new_zeros(
                (self.num_classes * K, *latents.shape[1:]), device='cpu')
            self._pool_feat = feats.new_zeros((self.num_classes * K, feats.shape[1]))

        for c in torch.unique(labels_cpu):
            ci = int(c)
            sel = (labels_cpu == c).nonzero(as_tuple=True)[0]     # rows in batch (CPU)
            nc = sel.numel()
            slots = (int(self._ptr[ci]) + torch.arange(nc)) % K   # ring within class block
            gidx = ci * K + slots                                 # global indices (CPU)
            sel_dev = sel.to(dev)
            self._pool[gidx] = latents[sel_dev].cpu()
            self._pool_feat[gidx.to(feats.device)] = feats[sel_dev]
            self._ptr[ci] = (int(self._ptr[ci]) + nc) % K
            self._filled[ci] = min(int(self._filled[ci]) + nc, K)
        self._all_idx_cache = None  # invalidate

    @torch.no_grad()
    def sample_bank_idx(self, labels, device):
        """Draw ``[num, bank_k]`` reservoir indices, one fixed bank per row,
        restricted to the row's class block (hard class-conditional). A class with
        no members yet falls back to the whole pool (unconditional). Reusable
        across ``x0_hat`` calls so a rollout trajectory keeps one bank per
        timestep."""
        labels = labels.to('cpu').long()
        num = labels.shape[0]
        out = torch.empty((num, self.bank_k), dtype=torch.long)
        for c in torch.unique(labels):
            rows = (labels == c).nonzero(as_tuple=True)[0]
            mem = self._class_indices(int(c))
            if mem is None:
                mem = self._all_indices()
            out[rows] = mem[torch.randint(0, mem.numel(), (rows.numel(), self.bank_k))]
        return out.to(device)

    @torch.no_grad()
    def x0_hat(self, x_t, sigma, feat_fn, bank_idx=None):
        """Posterior-mean data estimate over a sampled ``bank_k`` bank (rollout).

        Args:
            x_t (Tensor): ``[B, C, H, W]`` visited states (diffusion input space).
            sigma (float | Tensor): scalar or ``[B]`` noise levels in ``[0, 1]``.
            feat_fn (callable): ``[*, C, H, W] -> [*, Df]`` compact projection.
            bank_idx (Tensor | None): ``[B, bank_k]`` global reservoir indices
                (from :meth:`sample_bank_idx`); if ``None`` an unconditional bank
                is drawn from the whole pool.
        Returns:
            Tensor: ``[B, C, H, W]`` posterior-mean data ``x0_hat``.
        """
        assert self.ready, 'EmpiricalExpert reservoir is not filled yet.'
        B = x_t.shape[0]
        device = x_t.device
        if not torch.is_tensor(sigma):
            sigma = torch.full((B,), float(sigma), device=device)
        sigma = sigma.to(device).reshape(B)

        x_feat_all = feat_fn(x_t).flatten(1)  # [B, Df]
        out = torch.empty_like(x_t)

        for s in range(0, B, self.sample_chunk):
            e = min(s + self.sample_chunk, B)
            n = e - s
            sc = sigma[s:e].clamp_min(self.min_sigma)            # [n]
            x_feat = x_feat_all[s:e]                             # [n, Df]

            if bank_idx is not None:
                idx = bank_idx[s:e].to(device)
            else:
                all_idx = self._all_indices().to(device)
                idx = all_idx[torch.randint(0, all_idx.numel(), (n, self.bank_k), device=device)]
            feat_bank = self._pool_feat[idx]                    # [n, K, Df]

            # residual in subspace: feat(x_t) - (1 - sigma) feat(x_0^i)
            resid = x_feat.unsqueeze(1) - (1 - sc).view(n, 1, 1) * feat_bank
            d2 = resid.pow(2).sum(-1) / (2.0 * sc.view(n, 1) ** 2)  # [n, K]
            w = torch.softmax(-d2, dim=1)                          # [n, K] over the whole bank

            # weighted sum over the full bank, accumulated in blocks so only
            # sample_chunk x bank_chunk full-resolution latents are resident.
            acc = torch.zeros((n, *x_t.shape[1:]), device=device, dtype=out.dtype)
            for b0 in range(0, self.bank_k, self.bank_chunk):
                b1 = min(b0 + self.bank_chunk, self.bank_k)
                blk = b1 - b0
                full = self._pool[idx[:, b0:b1].reshape(-1).cpu()].to(device).view(
                    n, blk, *x_t.shape[1:])
                acc += (w[:, b0:b1].view(n, blk, *([1] * (x_t.dim() - 1))) * full).sum(1)
            out[s:e] = acc

        return out

    @torch.no_grad()
    def x0_hat_full(self, x_t, sigma, feat_fn, labels):
        """Posterior mean over the ENTIRE class-restricted pool (all same-class
        members), rather than a sampled ``bank_k`` bank -- used for on-path points,
        which are few and want the exact per-class posterior.

        Rows sharing a class share one member set, so the class's full latents are
        gathered once (blocked by ``bank_chunk``). The null/unconditional class
        (members = whole pool) is randomly capped to ``bank_k`` to stay tractable,
        since on-path labelling runs every step; real classes (<= per_class_pool
        members) are used in full.
        """
        assert self.ready, 'EmpiricalExpert reservoir is not filled yet.'
        B = x_t.shape[0]
        device = x_t.device
        if not torch.is_tensor(sigma):
            sigma = torch.full((B,), float(sigma), device=device)
        sigma = sigma.to(device).reshape(B)
        x_feat_all = feat_fn(x_t).flatten(1)  # [B, Df]
        labels_cpu = labels.to('cpu').long()
        out = torch.empty_like(x_t)

        for c in torch.unique(labels_cpu):
            rows = (labels_cpu == c).nonzero(as_tuple=True)[0]
            members = self._class_indices(int(c))
            if members is None:
                members = self._all_indices()  # unconditional (whole pool)
            if members.numel() > self.bank_k:  # cap the unconditional / huge case
                members = members[torch.randperm(members.numel())[:self.bank_k]]
            M = members.numel()

            rows_d = rows.to(device)
            sc = sigma[rows_d].clamp_min(self.min_sigma)        # [n_c]
            xf = x_feat_all[rows_d]                             # [n_c, Df]
            feat_m = self._pool_feat[members.to(device)]        # [M, Df]

            resid = xf.unsqueeze(1) - (1 - sc).view(-1, 1, 1) * feat_m.unsqueeze(0)
            d2 = resid.pow(2).sum(-1) / (2.0 * sc.view(-1, 1) ** 2)  # [n_c, M]
            w = torch.softmax(-d2, dim=1)                           # [n_c, M]

            acc = torch.zeros((rows.numel(), *x_t.shape[1:]), device=device, dtype=out.dtype)
            for b0 in range(0, M, self.bank_chunk):
                b1 = min(b0 + self.bank_chunk, M)
                # one shared member block for all rows of this class: [1, blk, ...]
                full = self._pool[members[b0:b1].cpu()].to(device).view(
                    1, b1 - b0, *x_t.shape[1:])
                acc += (w[:, b0:b1].view(rows.numel(), b1 - b0, *([1] * (x_t.dim() - 1))) * full).sum(1)
            out[rows_d] = acc

        return out
