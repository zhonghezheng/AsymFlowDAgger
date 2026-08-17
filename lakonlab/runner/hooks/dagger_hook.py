# Copyright (c) 2026 Hansheng Chen

import gc

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

    Rollout is *unguided* (no CFG) by design: each trajectory is generated either
    class-conditionally (its sampled class, no guidance) or unconditionally (the
    CFG-dropout rows, class ``null``), so every visited state stays on-policy for
    the field that will be trained on it, and is labelled by a matching empirical
    bank (``sample_bank_idx(train_labels)`` -- class-restricted for conditional
    rows, whole-pool posterior for null rows). ``guidance_scale`` is guarded to
    ``1.0``; CFG-guided rollouts would be off-policy for the labelled velocity.
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
        guidance_scale (float): must be ``1.0`` -- rollouts are unguided by design
            (class-conditional without CFG, or unconditional). Guarded in __init__.
        label_time_dropout (bool): if True, roll out ALWAYS class-conditionally and
            apply the CFG dropout per captured point at *label* time instead of per
            trajectory before generation. Same expected conditional/null proportions
            and targets as the default, but null points then draw states from the
            conditional (class-marginal) distribution rather than the unconditional
            one -- equivalent only where the two coincide (high sigma). Default False.
        class_sampling (str): ``'uniform'`` (default) draws rollout classes
            uniformly; ``'proportional'`` draws them from the empirical data class
            prior (``expert.sample_labels``), matching the on-path stream's
            (mildly imbalanced) class distribution.
        use_ema_rollout (bool): roll out with the EMA policy instead of online.
        aggregate_buffer (bool): accumulate rounds instead of rebuilding.
        sampler (str): scheduler name for rollout.
    """

    def __init__(self,
                 round_interval,
                 n_rollout,
                 nfe,
                 latent_size,
                 start_iter=0,
                 num_classes=1000,
                 null_label=1000,
                 prob_class=None,
                 rollout_chunk=128,
                 t_split=None,
                 guidance_scale=1.0,
                 label_time_dropout=False,
                 class_sampling='uniform',
                 use_ema_rollout=False,
                 aggregate_buffer=False,
                 sampler='FlowHeunODE'):
        self.round_interval = round_interval
        self.start_iter = start_iter  # delay first rollout round -> optimizer/LR warmup on pure on-path FM
        self.n_rollout = n_rollout
        self.nfe = nfe
        self.latent_size = tuple(latent_size)
        self.num_classes = num_classes
        self.null_label = null_label
        self.prob_class = prob_class
        self.rollout_chunk = rollout_chunk
        self.t_split = t_split
        assert guidance_scale == 1.0, (
            'DaggerRolloutHook requires guidance_scale == 1.0: rollouts must be '
            'unguided (class-conditional without CFG, or unconditional) so the '
            'expert labels the states the trained field actually visits. A '
            'CFG-guided rollout would be off-policy for the labelled velocity.')
        self.guidance_scale = guidance_scale
        self.label_time_dropout = label_time_dropout
        assert class_sampling in ('uniform', 'proportional')
        self.class_sampling = class_sampling
        self.use_ema_rollout = use_ema_rollout
        self.aggregate_buffer = aggregate_buffer
        self.sampler = sampler

    @staticmethod
    def _unwrap(runner):
        return runner.model.module if is_module_wrapper(runner.model) else runner.model

    def before_train_iter(self, runner):
        # warmup: no rollout rounds until start_iter (pure on-path FM first, so the
        # optimizer settles at the new LR before the DAGGER stream is introduced).
        if self.round_interval <= 0 or runner.iter < self.start_iter \
                or runner.iter % self.round_interval != 0:
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
        # keep dropout inline with the on-path stream: default to the model's
        # train_cfg.prob_class (same keep-vs-null mechanism, same probability).
        prob_class = self.prob_class if self.prob_class is not None \
            else model.train_cfg.get('prob_class', 1.0)
        # rollout/expert covers the noise side sigma >= t_split; default to the
        # diffusion's t_split so it stays complementary to the on-path stream.
        t_split = self.t_split if self.t_split is not None \
            else getattr(diffusion, 't_split', None)

        # expert banks are built from real images on disk, encoded with the exact
        # train pipeline: patchify(vae.encode(img*2-1)) -> diffusion input space.
        feat_fn = diffusion.feat_fn
        vae = model.vae
        vae_dtype = vae.dtype if hasattr(vae, 'dtype') else next(vae.parameters()).dtype

        def encode_fn(imgs):
            lat = vae.encode((imgs * 2 - 1).to(vae_dtype)).float()
            return model.patchify(lat)

        if not self.aggregate_buffer:
            model.dagger_buffer = []

        c, h, w = self.latent_size
        remaining = self.n_rollout
        while remaining > 0:
            b = min(self.rollout_chunk, remaining)
            remaining -= b

            noise = model.patchify(torch.randn((b, c, h, w), device=device))
            if self.class_sampling == 'proportional':
                # match the dataset's (mildly imbalanced) class prior, like the
                # on-path stream, rather than sampling classes uniformly.
                labels = model.expert.sample_labels(b, device)
            else:
                labels = torch.randint(0, self.num_classes, (b,), device=device)  # uniform

            if self.label_time_dropout:
                # VARIANT: rollout is ALWAYS class-conditional; CFG dropout is applied
                # per captured point at LABEL time (a fresh mask each step relabels
                # points to null). NB: null points then come from the conditional
                # (class-marginal) state distribution, not the unconditional one.
                class_labels = labels
            else:
                # DEFAULT: trajectory-level CFG dropout applied BEFORE generation, so a
                # dropped trajectory is GENERATED unconditionally and stays on-policy
                # for the unconditional field. Generation / label / target all key on
                # train_labels.
                train_labels = labels
                if prob_class < 1.0:
                    keep = torch.rand(b, device=device) < prob_class
                    train_labels = torch.where(
                        keep, labels, torch.full_like(labels, self.null_label))
                class_labels = train_labels

            # per-trajectory banks: each row gets its OWN conditional (drawn from its
            # sampled class) + null (whole-dataset) bank. x0_hat picks cond/null per
            # captured point via point_labels; no per-class or shared-null pooling.
            cond_banks, null_banks = model.expert.build_banks(labels, encode_fn, feat_fn, device)

            test_cfg_override = dict(sampler=self.sampler, num_timesteps=self.nfe)

            captured = []

            def sample_callback(net_self, kw):
                sigma = float(kw['t']) / diffusion.num_timesteps
                if t_split is None or sigma >= t_split:
                    x_t = kw['x_t']
                    if self.label_time_dropout:
                        # fresh per-point CFG-dropout mask at label time
                        if prob_class < 1.0:
                            keep = torch.rand(b, device=device) < prob_class
                        else:
                            keep = torch.ones(b, dtype=torch.bool, device=device)
                        point_labels = torch.where(
                            keep, labels, torch.full_like(labels, self.null_label))
                    else:
                        point_labels = train_labels
                    x0_hat = model.expert.x0_hat(
                        x_t, sigma, feat_fn, point_labels, cond_banks, null_banks, self.null_label)
                    captured.append((x_t.detach().cpu(), sigma,
                                     x0_hat.detach().cpu(), point_labels.detach().cpu()))
                return kw

            net.forward_test(
                noise=noise,
                guidance_scale=1.0,  # unguided (guarded in __init__)
                test_cfg_override=test_cfg_override,
                class_labels=class_labels,
                sample_callback=sample_callback)

            model.dagger_buffer.extend(captured)
            # free this chunk's banks promptly (they can be tens of GB with entire
            # banks x 4 ranks); gc.collect breaks any closure cycle holding them.
            del cond_banks, null_banks
            gc.collect()

        if was_training:
            diffusion.train()
