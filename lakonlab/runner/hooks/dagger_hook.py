# Copyright (c) 2026 Hansheng Chen

import torch

from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class DaggerRolloutHook(Hook):
    """Periodically roll the policy out, expert-label the visited states, and
    (re)fill the model's DAGGER replay buffer.

    Every ``round_interval`` iterations this rolls ``n_rollout`` samples from
    noise toward data with the current policy (reusing ``forward_test``'s
    ``sample_callback`` to capture each visited ``(x_t, sigma)``), labels them
    with the empirical expert's posterior-mean ``x0_hat``, and stores
    ``(x_t, sigma, x0_hat, label)`` in ``model.dagger_buffer``. The training step
    then mixes those in via ``LatentDiffusionClassImageDagger``.

    Rollout is class-*conditional*: each trajectory draws a random class, is
    integrated with the inference CFG (``guidance_scale`` / ``guidance_interval``,
    matching eval so visited states match the true sampling distribution), and is
    labelled by a class-restricted empirical bank (``sample_bank_idx(labels)``).
    Under DDP each rank rolls out and labels independently (data-parallel). Runs
    in ``before_train_iter`` so the fresh buffer is available to the same
    iteration's train step.

    Memory: the buffer holds ``n_rollout * nfe`` latents (x_t and x0_hat) on CPU;
    scale ``n_rollout`` / ``nfe`` / ``t_split`` to your RAM. ``aggregate_buffer``
    keeps all rounds (classic DAgger union) instead of rebuilding each round.

    Args:
        round_interval (int): iterations between rollout rounds.
        n_rollout (int): trajectories rolled out per round.
        nfe (int): Euler steps (function evals) per rollout.
        latent_size (tuple): ``(C, H, W)`` of the diffusion input space.
        num_classes (int): number of classes to sample rollout labels from.
        null_label (int): null/negative class index for the CFG uncond branch.
        prob_class (float | None): keep-class probability for the DAGGER stream
            (CFG dropout). ``<1`` trains a fraction of trajectories as the null
            class against the unconditional posterior; ``1.0`` disables dropout.
            ``None`` (default) inherits the on-path ``train_cfg.prob_class`` so
            the two streams stay inline automatically.
        rollout_chunk (int): trajectories integrated at once (GPU memory bound).
        t_split (float | None): only capture visited states with ``sigma >=
            t_split`` (the noise side; complementary to the on-path stream, which
            covers ``sigma < t_split``). ``None`` (default) inherits
            ``diffusion.t_split``; if that is also ``None``, every step is kept.
        guidance_scale (float): CFG scale for rollout (1.0 = none).
        guidance_interval (list | None): ``[lo, hi]`` sigma window for CFG.
        use_ema_rollout (bool): roll out with the EMA policy instead of online.
        aggregate_buffer (bool): accumulate rounds instead of rebuilding.
        sampler (str): scheduler name for rollout.
    """

    def __init__(self,
                 round_interval,
                 n_rollout,
                 nfe,
                 latent_size,
                 num_classes=1000,
                 null_label=1000,
                 prob_class=None,
                 rollout_chunk=128,
                 t_split=None,
                 guidance_scale=1.0,
                 guidance_interval=None,
                 use_ema_rollout=False,
                 aggregate_buffer=False,
                 sampler='FlowEulerODE'):
        self.round_interval = round_interval
        self.n_rollout = n_rollout
        self.nfe = nfe
        self.latent_size = tuple(latent_size)
        self.num_classes = num_classes
        self.null_label = null_label
        self.prob_class = prob_class
        self.rollout_chunk = rollout_chunk
        self.t_split = t_split
        self.guidance_scale = guidance_scale
        self.guidance_interval = guidance_interval
        self.use_ema_rollout = use_ema_rollout
        self.aggregate_buffer = aggregate_buffer
        self.sampler = sampler

    @staticmethod
    def _unwrap(runner):
        return runner.model.module if is_module_wrapper(runner.model) else runner.model

    def before_train_iter(self, runner):
        if self.round_interval <= 0 or runner.iter % self.round_interval != 0:
            return
        model = self._unwrap(runner)
        # wait until the reservoir has enough real data to form a bank
        if model.expert is None or not model.expert.ready:
            return
        self._run_round(model)

    @torch.no_grad()
    def _run_round(self, model):
        # unwrap the DDP layer around self.diffusion so custom methods
        # (forward_test / feat_fn / num_timesteps ...) are reachable; the EMA
        # copy is never DDP-wrapped.
        diffusion = model.diffusion.module if is_module_wrapper(model.diffusion) \
            else model.diffusion
        was_training = diffusion.training
        diffusion.eval()
        net = model.diffusion_ema if (self.use_ema_rollout and model.diffusion_use_ema) \
            else diffusion

        device = next(diffusion.parameters()).device
        use_cfg = self.guidance_scale is not None and self.guidance_scale > 1.0
        # keep dropout inline with the on-path stream: default to the model's
        # train_cfg.prob_class (same keep-vs-null mechanism, same probability).
        prob_class = self.prob_class if self.prob_class is not None \
            else model.train_cfg.get('prob_class', 1.0)
        # rollout/expert covers the noise side sigma >= t_split; default to the
        # diffusion's t_split so it stays complementary to the on-path stream.
        t_split = self.t_split if self.t_split is not None \
            else getattr(diffusion, 't_split', None)

        if not self.aggregate_buffer:
            model.dagger_buffer = []

        c, h, w = self.latent_size
        remaining = self.n_rollout
        while remaining > 0:
            b = min(self.rollout_chunk, remaining)
            remaining -= b

            noise = model.patchify(torch.randn((b, c, h, w), device=device))
            labels = torch.randint(0, self.num_classes, (b,), device=device)  # sampled classes

            # CFG dropout for the DAGGER stream (analogue of on-path prob_class):
            # some trajectories are trained as the null/unconditional class so the
            # unconditional field -- which co-generates the CFG states -- also gets
            # corrected. State generation still uses the true class; dropout only
            # changes the training label + expert bank.
            train_labels = labels
            if prob_class < 1.0:
                keep = torch.rand(b, device=device) < prob_class
                train_labels = torch.where(
                    keep, labels, torch.full_like(labels, self.null_label))

            # null rows have no reservoir members -> sample_bank_idx falls back to
            # the full pool, i.e. the *unconditional* posterior mean.
            bank_idx = model.expert.sample_bank_idx(train_labels, device)  # pinned per trajectory

            # class conditioning; under CFG forward_test expects [negative, positive]
            if use_cfg:
                null = torch.full((b,), self.null_label, dtype=torch.long, device=device)
                class_labels = torch.cat([null, labels], dim=0)
            else:
                class_labels = labels

            test_cfg_override = dict(sampler=self.sampler, num_timesteps=self.nfe)
            if use_cfg and self.guidance_interval is not None:
                test_cfg_override['guidance_interval'] = self.guidance_interval

            captured = []

            def sample_callback(net_self, kw):
                sigma = float(kw['t']) / diffusion.num_timesteps
                if t_split is None or sigma >= t_split:
                    x_t = kw['x_t']
                    x0_hat = model.expert.x0_hat(
                        x_t, sigma, diffusion.feat_fn, bank_idx=bank_idx)
                    captured.append((x_t.detach().cpu(), sigma,
                                     x0_hat.detach().cpu(), train_labels.detach().cpu()))
                return kw

            net.forward_test(
                noise=noise,
                guidance_scale=self.guidance_scale if use_cfg else 1.0,
                test_cfg_override=test_cfg_override,
                class_labels=class_labels,
                sample_callback=sample_callback)

            model.dagger_buffer.extend(captured)

        if was_training:
            diffusion.train()
