# Copyright (c) 2026 Hansheng Chen

import inspect
from copy import deepcopy

import torch

from .builder import MODELS, build_module
from .latent_diffusion_class_image import LatentDiffusionClassImage
from lakonlab.utils import rgetattr


@MODELS.register_module()
class LatentDiffusionClassImageDagger(LatentDiffusionClassImage):
    """Class-conditional latent diffusion with a DAGGER rollout stream.

    Holds the empirical expert and the rollout replay buffer (kept here rather
    than on ``self.diffusion`` so they stay out of the EMA deepcopy / parameter
    tying). Each train minibatch grows the expert reservoir from the real
    training latents already in the batch (no extra IO), and injects a
    ``buffer_batch`` of expert-labelled rollout states into the diffusion's
    ``forward_train``. The buffer itself is (re)filled by ``DaggerRolloutHook``.

    Args:
        expert (dict | None): config for the :class:`EmpiricalExpert`.
        roll_batch_size (int | None): rollout rows per minibatch when NOT using a
            timestep split (added alongside the full on-path batch). ``None``
            matches the on-path batch size. Ignored when ``diffusion.t_split`` is
            set (the split then carves a fixed ``batch_size`` instead).
    """

    def __init__(self, *args, expert=None, roll_batch_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert = build_module(expert) if expert is not None else None
        self.roll_batch_size = roll_batch_size
        self.dagger_buffer = []  # list of (x_t, sigma, x0_hat, label) cpu tuples; filled by the hook
        # give the trainable diffusion a NON-registered handle to the expert (so it
        # can label on-path states when onpath_expert_vel is set) without making it
        # a submodule -- object.__setattr__ bypasses nn.Module registration, keeping
        # it out of the EMA deepcopy / DDP wrap / parameter tying.
        if self.expert is not None:
            object.__setattr__(self.diffusion, '_dagger_expert', self.expert)
        self._splice_cache = {}  # class -> (imgs_uint8_cpu [N,3,H,W], feats [2N, Df] device)

    # ---- inference-time expert-velocity splice (diagnostic) -------------------
    # For t > t_split use the empirical expert's conditional velocity toward the
    # per-class posterior mean; for t <= t_split use the model's own velocity.
    # Banks are the FULL class (bank_size=None), cached per class as uint8 images
    # (flips reconstructed on gather) + subspace features, so RAM stays ~dataset
    # size and averaging re-encodes on the fly (RGBColorEncoder is a cheap affine).

    def _expert_encode_fn(self, imgs):
        """Images ``[M, 3, H, W]`` in ``[0, 1]`` -> the diffusion input space
        (``patchify(vae.encode(img * 2 - 1))``), matching the train pipeline exactly.
        Used by diffusions that build empirical-expert banks inside ``forward_train``
        (GaussianFlowOnPolicy); the DaggerRolloutHook builds the same closure."""
        if hasattr(self.vae, 'dtype'):
            vae_dtype = self.vae.dtype
        else:
            vae_dtype = next(self.vae.parameters()).dtype
        return self.patchify(self.vae.encode((imgs * 2 - 1).to(vae_dtype)).float())

    def _splice_encode_fn(self, imgs):
        if hasattr(self.vae, 'dtype'):
            vae_dtype = self.vae.dtype
        else:
            vae_dtype = next(self.vae.parameters()).dtype
        lat = self.vae.encode((imgs * 2 - 1).to(vae_dtype)).float()
        return self.patchify(lat)

    @torch.no_grad()
    def _ensure_splice_banks(self, labels, feat_fn, device):
        """Build+cache the FULL-class bank (uint8 originals + both-orientation feats)
        for every class in ``labels`` not seen yet."""
        for c in labels.tolist():
            c = int(c)
            if c in self._splice_cache:
                continue
            paths = self.expert._draw_paths(c)            # bank_size=None -> entire class
            imgs = self.expert._load_images(paths, device)  # [N,3,H,W] in [0,1]
            imgs_both = torch.cat([imgs, torch.flip(imgs, dims=[-1])], dim=0)
            feats = feat_fn(self._splice_encode_fn(imgs_both)).flatten(1).detach().half()
            imgs_u = imgs.mul(255).round().clamp_(0, 255).to(torch.uint8).cpu()
            self._splice_cache[c] = (imgs_u, feats)

    @torch.no_grad()
    def _splice_x0_hat(self, x_t, sigma, feat_fn, labels, device):
        """Per-row posterior-mean data estimate over the row's FULL class bank,
        exact (no top-K) -- mirrors EmpiricalExpert.x0_hat but re-encodes the cached
        uint8 images (orig + h-flip) in chunks instead of holding latents."""
        B = x_t.shape[0]
        if not torch.is_tensor(sigma):
            sigma = torch.full((B,), float(sigma), device=device)
        sigma = sigma.to(device).reshape(B)
        x_feat_all = feat_fn(x_t).flatten(1)              # [B, Df]
        labels_cpu = labels.to('cpu').long()
        min_sigma = self.expert.min_sigma
        chunk = self.expert.bank_chunk
        out = torch.empty_like(x_t)
        for i in range(B):
            imgs_u, feat_m = self._splice_cache[int(labels_cpu[i])]
            N = imgs_u.shape[0]                            # originals; feat_m is [2N, Df]
            sc = sigma[i].clamp_min(min_sigma)
            xf = x_feat_all[i]
            resid = xf.unsqueeze(0) - (1 - sc) * feat_m.float()   # [2N, Df]
            d2 = resid.pow(2).sum(-1) / (2.0 * sc ** 2)           # [2N]
            w = torch.softmax(-d2, dim=0)                         # [2N]
            acc = torch.zeros(x_t.shape[1:], device=device, dtype=out.dtype)
            for half in (0, 1):                            # 0 = original, 1 = h-flip
                w_h = w[half * N:(half + 1) * N]
                for b0 in range(0, N, chunk):
                    b1 = min(b0 + chunk, N)
                    imgs = imgs_u[b0:b1].to(device).float().div_(255.0)
                    if half == 1:
                        imgs = torch.flip(imgs, dims=[-1])
                    lat = self._splice_encode_fn(imgs)     # [blk, C, H, W] (patchified)
                    acc += (w_h[b0:b1].view(-1, *([1] * (x_t.dim() - 1))) * lat).sum(0)
            out[i] = acc
        return out

    def _get_splice_callback(self, diffusion, feat_fn, labels, t_split, device):
        nts = diffusion.num_timesteps

        def cb(net_self, kw):
            sigma = float(kw['t']) / nts
            if sigma > t_split:
                x_t = kw['x_t']
                x0_hat = self._splice_x0_hat(x_t, sigma, feat_fn, labels, device)
                _, sigma_clamped, _ = diffusion.get_clamp_coef(t=kw['t'], x_t=x_t)
                kw['denoising_output'] = (x_t - x0_hat) / sigma_clamped
            return kw

        return cb

    def val_step(self, data, test_cfg_override=dict(), **kwargs):
        cfg = deepcopy(self.test_cfg)
        cfg.update(test_cfg_override)
        if not cfg.get('expert_splice', False) or self.expert is None:
            return super().val_step(data, test_cfg_override=test_cfg_override, **kwargs)

        bs = len(data['labels'])
        guidance_scale = cfg.get('guidance_scale', 1.0)
        diffusion = self.diffusion_ema if self.diffusion_use_ema else self.diffusion
        feat_fn = diffusion.feat_fn
        t_split = cfg.get('t_split', rgetattr(diffusion, 't_split', 0.92))

        with torch.no_grad():
            labels = data['labels']
            device = labels.device
            self._ensure_splice_banks(labels, feat_fn, device)

            class_labels = labels
            if guidance_scale == 0.0:
                class_labels = data['negative_labels']
            elif guidance_scale != 1.0:
                class_labels = torch.cat([data['negative_labels'], class_labels], dim=0)

            # expert splices in the CONDITIONAL velocity at high sigma, keyed on the
            # true class labels (guidance is off there under the default interval).
            kwargs = dict(
                class_labels=class_labels,
                sample_callback=self._get_splice_callback(
                    diffusion, feat_fn, labels, t_split, device))

            if 'noise' in data:
                noise = data['noise']
            else:
                noise = torch.randn((bs, *cfg['latent_size']), device=device)
            noise = self.patchify(noise)

            latents_out = diffusion(
                noise=noise,
                guidance_scale=guidance_scale,
                test_cfg_override=test_cfg_override,
                **kwargs)
            latents_out = self.unpatchify(latents_out)

            if hasattr(self.vae, 'dtype'):
                vae_dtype = self.vae.dtype
            else:
                vae_dtype = next(self.vae.parameters()).dtype
            out_images = (self.vae.decode(latents_out.to(vae_dtype)).float() / 2 + 0.5).clamp(0, 1)

        return dict(num_samples=bs, pred_imgs=out_images)

    def _prepare_train_minibatch_args(self, data, running_status=None):
        bs, diffusion_args, diffusion_kwargs = super()._prepare_train_minibatch_args(
            data, running_status)

        # (the expert's banks come from disk on demand -- no reservoir to grow here.)

        # The UNDROPPED class labels, for diffusions that need the true class rather
        # than the CFG-dropout-applied one (GaussianFlowOnPolicy's cfg-gap term
        # compares cond vs null per row, so a row already relabelled to null would
        # compare the unconditional branch against itself). Passed only when
        # forward_train declares it, so it never leaks into self.pred's kwargs.
        params = inspect.signature(
            rgetattr(self.diffusion, 'forward_train')).parameters
        if 'class_labels_true' in params:
            diffusion_kwargs['class_labels_true'] = data['labels']
        # the expert builds its banks from real images on disk and needs them in the
        # diffusion's input space; the VAE lives here, not on the diffusion, so hand
        # the encoder down (same closure the DaggerRolloutHook uses).
        # NB not gated on self.expert: the MMD arms have expert=None but still need
        # the encoder, because their target side now reads real images from disk.
        if 'expert_encode_fn' in params:
            diffusion_kwargs['expert_encode_fn'] = self._expert_encode_fn

        # no buffer yet (before the first rollout round): pure full-range on-path.
        if not self.dagger_buffer:
            return bs, diffusion_args, diffusion_kwargs

        t_split = rgetattr(self.diffusion, 't_split', None)
        if t_split is not None:
            # fixed-total carve: total stays bs. On-path (sigma < t_split) carries
            # the (1 - p_high) mass, rollout (sigma >= t_split) the p_high mass, so
            # n_on = round(bs * (1 - p_high)), n_roll = bs - n_on. Only the first
            # n_on dataloader rows are used on-path; the rest of the batch budget
            # goes to expert-labelled rollout states.
            # frac_on_path blends REAL data (on-path FM) into the high-sigma region:
            # of the n_high (>= t_split) budget, n_high_on are on-path and the rest
            # are expert-labelled rollout. n_on + n_high_on + n_roll = bs, so the
            # per-point weighting stays uniform (proportional roll_weight -> w=n_roll/bs).
            p_high = rgetattr(self.diffusion, 'high_sigma_fraction')()
            frac_on_path = rgetattr(self.diffusion, 'frac_on_path', 0.0)
            n_on = int(round(bs * (1 - p_high)))        # low-sigma on-path (< t_split)
            n_on = max(1, min(n_on, bs - 1))
            n_high = bs - n_on                           # high-sigma budget (>= t_split)
            n_high_on = int(round(n_high * frac_on_path))   # real-data FM mixed into it
            n_roll = n_high - n_high_on                  # expert-labelled rollout states
            n_onpath = n_on + n_high_on                  # total real-data on-path rows
            diffusion_args = tuple(
                a[:n_onpath] if torch.is_tensor(a) else a for a in diffusion_args)
            if torch.is_tensor(diffusion_kwargs.get('class_labels', None)):
                diffusion_kwargs['class_labels'] = diffusion_kwargs['class_labels'][:n_onpath]
            # how many on-path rows are the high-sigma (>= t_split) mix-in
            diffusion_kwargs['n_onpath_high'] = n_high_on
            diffusion_kwargs['buffer_batch'] = self._sample_buffer(diffusion_args[0], n_roll)
        else:
            # no split: keep the full on-path batch and add rollout rows on top.
            n = self.roll_batch_size or bs
            diffusion_kwargs['buffer_batch'] = self._sample_buffer(diffusion_args[0], n)

        return bs, diffusion_args, diffusion_kwargs

    @torch.no_grad()
    def _sample_buffer(self, ref, n):
        """Draw exactly ``n`` expert-labelled rollout rows from the buffer.

        Each row independently picks a buffer entry (hence its own ``sigma`` and
        class ``label``) and a random row within that entry, mirroring the
        reference implementation. ``ref`` provides device/latent-shape only."""
        buf = self.dagger_buffer
        if not buf or n <= 0:
            return None
        device = ref.device
        j = torch.randint(0, len(buf), (n,))
        x_t = ref.new_empty((n, *ref.shape[1:]))
        sigma = ref.new_empty((n,))
        x0_hat = ref.new_empty((n, *ref.shape[1:]))
        labels = torch.empty((n,), dtype=torch.long, device=device)
        # present only when DaggerRolloutHook ran with store_cfg_targets (7-tuples)
        has_cfg = len(buf[0]) >= 7
        if has_cfg:
            x0_cond = ref.new_empty((n, *ref.shape[1:]))
            x0_null = ref.new_empty((n, *ref.shape[1:]))
            labels_true = torch.empty((n,), dtype=torch.long, device=device)
        for jj in torch.unique(j):
            rows = (j == jj).nonzero(as_tuple=True)[0]
            entry = buf[int(jj)]
            bx, bsig, bv, blab = entry[:4]
            ri = torch.randint(0, bx.size(0), (rows.numel(),))
            x_t[rows] = bx[ri].to(device)
            sigma[rows] = float(bsig)
            x0_hat[rows] = bv[ri].to(device)
            labels[rows] = blab[ri].to(device)
            if has_cfg:
                bc, bn, btrue = entry[4:7]
                x0_cond[rows] = bc[ri].to(device)
                x0_null[rows] = bn[ri].to(device)
                labels_true[rows] = btrue[ri].to(device)
        out = dict(x_t=x_t, sigma=sigma, x0_hat=x0_hat, labels=labels)
        if has_cfg:
            out.update(x0_cond=x0_cond, x0_null=x0_null, labels_true=labels_true)
        return out
