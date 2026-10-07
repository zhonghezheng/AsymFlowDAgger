# Copyright (c) 2026 Hansheng Chen

import os.path as osp

import torch

from .experts import EmpiricalExpert

_U8_DEFAULT = '/dev/shm/asymflow/train_u8_256'


class EmpiricalVelocity:
    """Inference-time velocity override: for ``sigma_switch < sigma <= sigma_hi`` the
    sampler's velocity is REPLACED by the empirical expert's conditional velocity

        v*_cond = (x_t - x0_hat) / sigma,    x0_hat = EmpiricalExpert.x0_hat over the
                                             row's ENTIRE class bank (x2 flips)

    and outside that window the model (with CFG) runs unchanged -- above it when
    ``sigma_hi < 1`` (default 1: v* from the very first step). It is the expert the
    band terms train on (bank_size=None, include_flips, feat kernel, optional
    temp_spread), so this measures what the band target does when it DRIVES the
    trajectory instead of being regressed onto.

    Driven by ``GaussianFlow.forward_test(sample_callback=...)``, which calls it after
    every network eval (CFG already combined). With the run default the CFG interval
    is [0, sigma_switch], so above the switch the eval is the plain conditional
    velocity and the override replaces exactly v_cond.

    ``complete_step`` (Heun): the step that LANDS on the switch, sigma_k > switch ->
    sigma_{k+1} <= switch, also takes v* at its corrector eval, so the whole segment
    1 -> sigma_switch is integrated with v* and the model starts from the predictor
    at the switch state. A Heun corrector is the one eval whose t is lower than the
    previous call's; that is how it is recognised. Leave it off for one-eval-per-step
    samplers (Euler), where that test would also catch the next predictor. The upper
    edge is cut the same way, by step: the step STARTING at sigma_hi is v*'s (both of
    its evals), the one landing on it is the model's (its corrector included), so
    v* integrates exactly sigma_hi -> sigma_switch.

    Banks are built per eval batch with :meth:`EmpiricalExpert.build_banks` (the
    u8 cache when present), one per distinct class, held as uint8 on the host and
    streamed to the GPU per eval, then dropped at :meth:`end`.

    Logged (``end``, averaged by the eval over batches and ranks):
      ``ev_reldiff``        ||v* - v_model|| / ||v_model|| over the overridden evals
                            (v_model is the eval the override discards);
      ``ev_reldiff_s<s>``   the same at the first and last overridden sigma;
      ``ev_ess`` / ``ev_wmax``  posterior effective sample size and max weight at the
                            last overridden eval (feat kernel only) -- ESS ~1 means the
                            trajectory has been handed to ONE training image.
    """

    def __init__(self,
                 sigma_switch=0.88,
                 sigma_hi=1.0,
                 temp_spread=None,
                 kernel_space='feat',
                 include_flips=True,
                 complete_step=True,
                 datalist='data/imagenet/train.txt',
                 data_root='data/imagenet/train/',
                 u8_cache='auto',
                 u8_read_threads=8,
                 bank_chunk=1024,
                 num_classes=1000):
        if u8_cache == 'auto':
            u8_cache = _U8_DEFAULT if osp.exists(_U8_DEFAULT + '.complete') else None
        self.sigma_switch = float(sigma_switch)
        self.sigma_hi = float(sigma_hi)
        assert self.sigma_hi > self.sigma_switch, (sigma_switch, sigma_hi)
        self.complete_step = bool(complete_step)
        self.expert = EmpiricalExpert(
            datalist_path=datalist, data_root=data_root, num_classes=num_classes,
            bank_size=None,                 # the entire class
            null_bank_size=1, null_bank_mode='shared',   # no null rows; keep its draw trivial
            include_flips=include_flips, kernel_space=kernel_space,
            temp_spread=(None if temp_spread is None else float(temp_spread)),
            # pread threads per process: a GPFS-resident cache is latency-bound per
            # read, so more in flight helps there (a /dev/shm one tops out at ~8)
            u8_cache=u8_cache, u8_read_threads=int(u8_read_threads), host_buffers=0)
        self.expert.bank_chunk = int(bank_chunk)
        self._labels = None
        self._banks = None
        self._feat_fn = None
        self._prev_t = None
        self._stats = None

    @staticmethod
    def _make_feat_fn(diffusion):
        """GaussianFlowDagger.feat_fn for any AsymFlow denoiser: rank-8 subspace
        coordinates ``[B, n_tokens, basis_rank]``."""
        p = diffusion.denoising

        def feat_fn(z):
            proj = p.proj_buffer.to(dtype=z.dtype, device=z.device)
            return p.pack(p.patchify(z, p.patch_size)) @ proj
        return feat_fn

    def begin(self, diffusion, labels, encode_fn):
        """Build this batch's class banks. ``labels`` are the rows' (real) classes,
        ``encode_fn`` maps [0, 1] images to the diffusion input space."""
        assert int(labels.max()) < self.expert.num_classes and int(labels.min()) >= 0, \
            'EmpiricalVelocity needs real class labels (not the null label)'
        self._feat_fn = self._make_feat_fn(diffusion)
        self._labels = labels
        cond_banks, _ = self.expert.build_banks(
            labels, encode_fn, self._feat_fn, labels.device)
        self._banks = cond_banks
        self._prev_t = None
        self._stats = dict(sigma=[], reldiff=[], ess=None, wmax=None)

    def end(self):
        self._banks = None
        self._labels = None
        st = self._stats
        out = dict()
        if not st or not st['sigma']:
            return out
        rd = torch.tensor(st['reldiff'])
        out['ev_reldiff'] = float(rd.mean())
        out[f'ev_reldiff_s{st["sigma"][0]:.2f}'] = st['reldiff'][0]
        out[f'ev_reldiff_s{st["sigma"][-1]:.2f}'] = st['reldiff'][-1]
        out['ev_evals'] = float(len(st['sigma']))
        if st['ess'] is not None:
            out['ev_ess'] = st['ess']
            out['ev_wmax'] = st['wmax']
        return out

    @torch.no_grad()
    def _ess(self, x_t, sigma):
        """ESS and max posterior weight per row, with x0_hat's own feat-kernel weights
        (same distances, adaptive temperature and clamp) -- mean over rows."""
        ex = self.expert
        sc = max(sigma, ex.min_sigma)
        feats = self._feat_fn(x_t.float()).flatten(1)
        ess, wmax = [], []
        for i, bank in enumerate(self._banks):
            fm = bank.feats if hasattr(bank, 'feats') else bank[1]
            d2 = (feats[i].unsqueeze(0) - (1 - sc) * fm).pow(2).sum(-1) / (2.0 * sc ** 2)
            if ex.temp_spread is not None and d2.numel() > 1:
                d2 = d2 / (d2.std() / ex.temp_spread).clamp_min(1.0)
            w = torch.softmax(-d2, dim=0)
            ess.append(float(1.0 / w.pow(2).sum()))
            wmax.append(float(w.max()))
        return sum(ess) / len(ess), sum(wmax) / len(wmax)

    @torch.no_grad()
    def __call__(self, diffusion, outputs):
        t = float(outputs['t'])
        sigma = t / diffusion.num_timesteps
        prev, self._prev_t = self._prev_t, t
        # each eval belongs to the step STARTING at sigma_k: its own t for a predictor,
        # the previous call's for a Heun corrector. The step is v*'s iff
        # sigma_switch < sigma_k <= sigma_hi (sigma_hi = 1 reduces to the original
        # 'above the switch, plus the corrector landing on it').
        corrector = self.complete_step and prev is not None and t < prev - 1e-9
        start = (prev if corrector else t) / diffusion.num_timesteps
        if not (self.sigma_switch + 1e-6 < start <= self.sigma_hi + 1e-6):
            return dict()
        x_t = outputs['x_t']
        v_model = outputs['denoising_output']
        x0_hat = self.expert.x0_hat(
            x_t.float(), sigma, self._feat_fn, self._labels, self._banks, self._banks,
            null_label=self.expert.num_classes)
        v = (x_t.float() - x0_hat) / max(sigma, 1e-6)
        dims = tuple(range(1, v.dim()))
        rel = ((v - v_model.float()).pow(2).sum(dims).sqrt()
               / v_model.float().pow(2).sum(dims).sqrt().clamp_min(1e-12))
        self._stats['sigma'].append(sigma)
        self._stats['reldiff'].append(float(rel.mean()))
        if self.expert.kernel_space == 'feat':
            # overwritten every eval, so end() reports the LAST overridden one
            self._stats['ess'], self._stats['wmax'] = self._ess(x_t, sigma)
        return dict(denoising_output=v.to(v_model.dtype))
