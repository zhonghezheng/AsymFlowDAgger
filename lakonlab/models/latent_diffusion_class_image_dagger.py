# Copyright (c) 2026 Hansheng Chen

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

    def _prepare_train_minibatch_args(self, data, running_status=None):
        bs, diffusion_args, diffusion_kwargs = super()._prepare_train_minibatch_args(
            data, running_status)

        # (the expert's banks come from disk on demand -- no reservoir to grow here.)

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
        for jj in torch.unique(j):
            rows = (j == jj).nonzero(as_tuple=True)[0]
            bx, bsig, bv, blab = buf[int(jj)]
            ri = torch.randint(0, bx.size(0), (rows.numel(),))
            x_t[rows] = bx[ri].to(device)
            sigma[rows] = float(bsig)
            x0_hat[rows] = bv[ri].to(device)
            labels[rows] = blab[ri].to(device)
        return dict(x_t=x_t, sigma=sigma, x0_hat=x0_hat, labels=labels)
