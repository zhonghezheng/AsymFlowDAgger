# Copyright (c) 2026 Hansheng Chen

import torch

from ..builder import MODULES
from .gaussian_flow import GaussianFlow
from lakonlab.models.architectures.utils import get_module_device


@MODULES.register_module()
class GaussianFlowDagger(GaussianFlow):
    """GaussianFlow with an added DAGGER stream.

    Matches the released ``asymflow_h_16_r8`` model: a plain ``GaussianFlow``
    velocity-MSE objective whose *network* (``AsymJiT``) carries the asymmetric
    low-rank parametrisation (``proj_buffer`` / calibration). On-path minibatches
    run the unchanged GaussianFlow loss (:meth:`GaussianFlow.forward_train`); when
    a ``buffer_batch`` of expert-labelled rollout states is supplied, a velocity
    term toward the empirical-expert target (through the *same* ``flow_loss``
    module) is mixed in as a CONVEX combination
    ``loss = (1 - w) * mean_onpath + w * mean_rollout``. ``roll_weight='proportional'``
    sets ``w = n_roll / bs`` -- the rollout point fraction from the t_split carve --
    so the combined loss is a single per-point mean over the whole batch and the
    sigma marginal matches the base schedule; a float uses a fixed convex weight.

    Expert target ``L`` (``complement_mode='project'``, the derived asym loss):

        L = (P x_t - (1 - sigma) P x0_hat - sigma x0_hat) / sigma_clamped

    which agrees with the shipped velocity in the rank-``basis_rank`` subspace and
    equals ``-x0_hat_comp`` in the orthogonal complement (low-rank prior).
    ``complement_mode='full'`` uses ``(x_t - x0_hat) / sigma_clamped`` (the plain
    velocity toward the posterior mean, complement kept). Both equal the shipped
    ``u_t`` on-path when ``x0_hat = x_0`` -- ``'full'`` exactly, ``'project'`` in
    the subspace.

    The DAGGER stream is *unguided* (no CFG): rollout states are generated either
    class-conditionally (their sampled class) or unconditionally (the CFG-dropout
    rows) and labelled by a matching empirical expert (class-restricted bank for
    conditional rows, whole-pool posterior for null rows); the buffer ``pred``
    conditions on those same (dropout-applied) labels, so each state is on-policy
    for the field trained on it. The on-path loss is class-conditional as usual.
    ``'project'`` mode requires an ``AsymJiT``-style denoising network exposing
    ``proj_buffer`` / ``pack`` / ``patchify``.

    ``onpath_expert_vel=True`` additionally relabels the *on-path* targets with
    the empirical expert's posterior-mean velocity (same ``expert_target`` /
    ``complement_mode``) instead of the true FM residual ``noise - x_0``. On-path
    points use the expert over the ENTIRE class-restricted pool (``x0_hat_full``,
    exact per-class posterior), whereas rollout trajectories use one sampled
    ``bank_k`` bank each. It needs the expert handle ``_dagger_expert`` set by the
    wrapper and is a no-op until the reservoir is ready.
    """

    def __init__(self,
                 *args,
                 complement_mode='project',
                 roll_weight='proportional',
                 t_split=None,
                 onpath_expert_vel=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert complement_mode in ('project', 'full')
        assert self.denoising_mean_mode.upper() == 'U', \
            'GaussianFlowDagger expects denoising_mean_mode="U".'
        if complement_mode == 'project':
            assert hasattr(self.denoising, 'proj_buffer'), \
                "complement_mode='project' needs an AsymJiT-style denoising with proj_buffer."
        # convex mix weight for the rollout stream: 'proportional' -> w = n_roll/bs
        # (matches the t_split carve, so on-path + rollout form one per-point mean);
        # or a fixed float convex weight in [0, 1].
        assert roll_weight == 'proportional' or (
            isinstance(roll_weight, (int, float)) and 0.0 <= roll_weight <= 1.0), \
            "roll_weight must be 'proportional' or a float convex weight in [0, 1]."
        self.complement_mode = complement_mode
        self.roll_weight = roll_weight
        # timestep split: on-path FM covers sigma < t_split (data side), the DAGGER
        # rollout/expert stream covers sigma >= t_split (noise side). None -> no split
        # (on-path over the full range, buffer over whatever the hook captures).
        self.t_split = t_split
        # if True, label on-path samples with the empirical expert's posterior-mean
        # velocity (using the same expert_target / complement_mode) instead of the
        # true FM residual noise - x_0. Requires the expert handle set by the wrapper.
        self.onpath_expert_vel = onpath_expert_vel
        self._p_high = None  # cached logit-normal mass at sigma >= t_split
        # non-registered handle to the wrapper's EmpiricalExpert (set via
        # object.__setattr__ so it is NOT a submodule -> stays out of EMA/DDP/tying).
        self._dagger_expert = None

    @torch.no_grad()
    def high_sigma_fraction(self, n_mc=100000):
        """Fraction of the on-path timestep distribution with ``sigma >= t_split``,
        estimated once by Monte Carlo from the model's own ``timestep_sampler``
        (robust to logit-normal / shift / warp). Used to proportion the two
        streams so the combined sigma marginal matches the base schedule."""
        assert self.t_split is not None
        if self._p_high is None:
            t = self.timestep_sampler(n_mc, device='cpu')
            sigma = t / self.num_timesteps
            frac = (sigma >= self.t_split).float().mean().item()
            self._p_high = float(min(max(frac, 1e-4), 1 - 1e-4))
        return self._p_high

    def _sample_t_below_split(self, num_batches, seq_len, device):
        """On-path timesteps drawn from the base sampler but truncated (by
        rejection) to ``sigma < t_split`` -- i.e. the base logit-normal restricted
        to the data side, so the on-path weighting is preserved."""
        out = torch.empty(num_batches, device=device)
        filled = 0
        while filled < num_batches:
            t = self.timestep_sampler(num_batches, seq_len=seq_len, device=device)
            ok = t[(t / self.num_timesteps) < self.t_split]
            take = min(num_batches - filled, ok.numel())
            if take > 0:
                out[filled:filled + take] = ok[:take]
                filled += take
        return out

    def feat_fn(self, z):
        """Compact rank-``basis_rank`` subspace features for expert scoring:
        ``[B, C, H, W] -> [B, n_tokens, basis_rank]``."""
        p = self.denoising
        proj = p.proj_buffer.to(dtype=z.dtype, device=z.device)  # (patch_dim, basis_rank)
        packed = p.pack(p.patchify(z, p.patch_size))             # (B, n_tokens, patch_dim)
        return packed @ proj                                     # (B, n_tokens, basis_rank)

    def project_fn(self, z):
        """Full orthogonal-subspace projection ``P z = z B Bt`` in latent space:
        ``[B, C, H, W] -> [B, C, H, W]``."""
        p = self.denoising
        proj = p.proj_buffer.to(dtype=z.dtype, device=z.device)  # (patch_dim, basis_rank)
        _, _, h, w = z.shape
        packed = p.pack(p.patchify(z, p.patch_size))             # (B, n_tokens, patch_dim)
        sub = packed @ proj @ proj.T                             # (B, n_tokens, patch_dim)
        return p.unpatchify(
            p.unpack(sub, h // p.patch_size, w // p.patch_size), p.patch_size)

    def expert_target(self, x_t, sigma, x0_hat):
        """Expert velocity target ``L`` at visited states (already ``/sigma_clamped``,
        i.e. in the same clamp-weighted velocity space as GaussianFlow's ``u_t``).

        'project' uses the single-projection factoring
        ``P(x_t - (1-t) x0_hat)/sigma_clamped - clamp_coef * x0_hat``;
        the clamp is folded into both terms to stay consistent with GaussianFlow's
        ``clamp_coef`` weighting near sigma->0)."""
        sig = sigma.reshape(x_t.size(0), *([1] * (x_t.dim() - 1))).to(x_t)
        _, sigma_clamped, clamp_coef = self.get_clamp_coef(sigma=sig, x_t=x_t)
        if self.complement_mode == 'project':
            return self.project_fn(x_t - (1 - sig) * x0_hat) / sigma_clamped \
                - clamp_coef * x0_hat
        else:  # 'full'
            return (x_t - x0_hat) / sigma_clamped

    def _onpath_loss(self, x_0, truncate, use_expert, **kwargs):
        """On-path GaussianFlow loss with optional sigma-truncation (``sigma <
        t_split``) and optional empirical-expert velocity labelling (mirrors
        GaussianFlow.forward_train)."""
        assert self.repa_loss is None, \
            't_split / onpath_expert_vel are not supported together with repa_loss.'
        device = get_module_device(self)
        num_batches = x_0.size(0)
        seq_len = x_0.shape[2:].numel()
        eps = self.train_cfg.get('eps', 1e-4)
        if truncate:
            t = self._sample_t_below_split(num_batches, seq_len, device)
        else:
            min_raw_t = self.train_cfg.get('min_raw_t', 0.0)
            max_raw_t = self.train_cfg.get('max_raw_t', 1.0)
            t = self.timestep_sampler(
                num_batches, seq_len=seq_len, device=device,
                raw_t_range=(min_raw_t, max_raw_t))
        t = t.clamp(min=eps, max=self.num_timesteps)
        noise = torch.randn_like(x_0)
        x_t, _, _ = self.sample_forward_diffusion(x_0, t, noise)
        denoising_output = self.pred(x_t, t, **kwargs)

        if use_expert:
            # label on-path states with the expert over the ENTIRE class-restricted
            # pool (exact per-class posterior; null/dropout rows -> capped full pool),
            # keyed on the on-path conditioning labels.
            sigma = t / self.num_timesteps
            labels = kwargs['class_labels']
            x0_hat = self._dagger_expert.x0_hat_full(x_t, sigma, self.feat_fn, labels)
            _, _, clamp_coef = self.get_clamp_coef(t=t, x_t=x_t)
            u_t_pred = denoising_output * clamp_coef
            u_t = self.expert_target(x_t, sigma, x0_hat)
            loss_diffusion = self.flow_loss(dict(u_t_pred=u_t_pred, u_t=u_t, timesteps=t))
        else:
            loss_diffusion = self.loss(denoising_output, x_0, noise, x_t, t)

        log_vars = self.flow_loss.log_vars.copy()
        log_vars.update(loss_diffusion=loss_diffusion.detach())
        return loss_diffusion, log_vars

    def forward_train(
            self,
            x_0,
            visual_encoder_features=None,
            running_status=None,
            buffer_batch=None,
            **kwargs):
        # On-path GaussianFlow velocity-MSE loss. A split (t_split with an active
        # buffer) restricts on-path to sigma < t_split; onpath_expert_vel relabels
        # on-path targets with the expert. Either path uses the reimplemented
        # _onpath_loss; otherwise defer to the base full-range implementation.
        use_split = self.t_split is not None and buffer_batch is not None
        expert = self._dagger_expert
        use_expert = (self.onpath_expert_vel and expert is not None
                      and expert.ready and 'class_labels' in kwargs)
        if use_split or use_expert:
            loss, log_vars = self._onpath_loss(
                x_0, truncate=use_split, use_expert=use_expert, **kwargs)
        else:
            loss, log_vars = super().forward_train(
                x_0,
                visual_encoder_features=visual_encoder_features,
                running_status=running_status,
                **kwargs)

        if buffer_batch is not None and self.roll_weight != 0:
            x_t = buffer_batch['x_t']
            sigma = buffer_batch['sigma'].reshape(-1)   # [Bb]
            x0_hat = buffer_batch['x0_hat']
            labels = buffer_batch['labels']             # class-conditional (the rollout classes)
            t = self.num_timesteps * sigma

            denoising_output = self.pred(x_t, t, class_labels=labels)
            sig = sigma.reshape(x_t.size(0), *([1] * (x_t.dim() - 1))).to(x_t)
            _, _, clamp_coef = self.get_clamp_coef(sigma=sig, x_t=x_t)
            u_t_pred = denoising_output * clamp_coef                 # denoising_mean_mode == 'U'
            u_t = self.expert_target(x_t, sigma, x0_hat)
            roll_loss = self.flow_loss(dict(u_t_pred=u_t_pred, u_t=u_t, timesteps=t))

            # convex combination weighted by the rollout point fraction. With the
            # t_split carve the on-path stream holds n_on = x_0.size(0) points and
            # the rollout holds n_roll = x_t.size(0) (n_on + n_roll = bs), so
            # 'proportional' (w = n_roll/bs) makes loss a single per-point mean over
            # the whole batch; a float uses a fixed convex weight instead.
            n_roll = x_t.size(0)
            n_on = x_0.size(0)
            w = n_roll / (n_on + n_roll) if self.roll_weight == 'proportional' \
                else float(self.roll_weight)
            loss = (1.0 - w) * loss + w * roll_loss
            log_vars['loss_dagger'] = roll_loss.detach()
            log_vars['roll_weight'] = roll_loss.new_tensor(w)

        return loss, log_vars
