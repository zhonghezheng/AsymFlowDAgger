# Copyright (c) 2026 Hansheng Chen

import torch
import torch.distributed as dist

from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook, get_dist_info


@HOOKS.register_module()
class WandbTrajectoryHook(Hook):
    """Log a denoising-trajectory image grid to wandb during evaluation.

    Every ``interval`` iterations (rank 0 only), samples ``n_samples`` images with
    the (EMA) policy using the inference sampler + CFG, capturing the model's
    predicted clean image ``x0`` at each of the ``nfe`` sampling steps, and logs
    the noise->data progression to wandb as ``tag`` (rows = samples, cols =
    steps). Requires an active wandb run (i.e. a ``WandbLoggerHook`` in
    ``log_config``); it is a no-op otherwise. The image is logged with
    ``commit=False`` so it attaches to the current wandb step without clashing
    with the scalar logger's step counter.

    Args:
        interval (int): iterations between trajectory logs.
        latent_size (tuple): ``(C, H, W)`` of the diffusion input space.
        nfe (int): sampling steps for the rollout.
        n_samples (int): number of trajectories (grid rows).
        max_cols (int): cap on trajectory columns; the captured states are
            evenly subsampled to this many (some samplers, e.g. Heun, fire the
            callback more than nfe times, so this keeps the grid readable).
        num_classes (int): classes to sample conditioning labels from.
        null_label (int): null class for the CFG uncond branch.
        guidance_scale (float): CFG scale (1.0 = none).
        guidance_interval (list | None): ``[lo, hi]`` sigma window for CFG.
        use_ema (bool): sample with the EMA policy.
        sampler (str): scheduler name.
        tag (str): wandb key for the logged image.
    """

    def __init__(self,
                 interval,
                 latent_size,
                 nfe=8,
                 n_samples=4,
                 max_cols=8,
                 num_classes=1000,
                 null_label=1000,
                 guidance_scale=1.0,
                 guidance_interval=None,
                 use_ema=True,
                 sampler='FlowEulerODE',
                 tag='rollout/denoising'):
        self.interval = interval
        self.latent_size = tuple(latent_size)
        self.nfe = nfe
        self.n_samples = n_samples
        self.max_cols = max_cols
        self.num_classes = num_classes
        self.null_label = null_label
        self.guidance_scale = guidance_scale
        self.guidance_interval = guidance_interval
        self.use_ema = use_ema
        self.sampler = sampler
        self.tag = tag

    def after_train_iter(self, runner):
        if self.interval <= 0 or not self.every_n_iters(runner, self.interval):
            return
        rank, ws = get_dist_info()
        try:
            import wandb
        except ImportError:
            wandb = None
        if rank == 0 and wandb is not None and wandb.run is not None:
            self._log_trajectory(runner, wandb)
        if ws > 1:
            dist.barrier()  # keep ranks in sync (only rank 0 sampled)

    @torch.no_grad()
    def _log_trajectory(self, runner, wandb):
        from torchvision.utils import make_grid

        model = runner.model.module if is_module_wrapper(runner.model) else runner.model
        # EMA copy is never DDP-wrapped; the online diffusion is, so unwrap it.
        diffusion = model.diffusion_ema if (self.use_ema and model.diffusion_use_ema) \
            else (model.diffusion.module if is_module_wrapper(model.diffusion)
                  else model.diffusion)
        was_training = diffusion.training
        diffusion.eval()
        device = next(diffusion.parameters()).device

        n = self.n_samples
        c, h, w = self.latent_size
        labels = torch.randint(0, self.num_classes, (n,), device=device)
        noise = model.patchify(torch.randn((n, c, h, w), device=device))

        use_cfg = self.guidance_scale is not None and self.guidance_scale > 1.0
        if use_cfg:
            null = torch.full((n,), self.null_label, dtype=torch.long, device=device)
            class_labels = torch.cat([null, labels], dim=0)  # forward_test expects [neg, pos]
        else:
            class_labels = labels
        test_cfg_override = dict(sampler=self.sampler, num_timesteps=self.nfe)
        if use_cfg and self.guidance_interval is not None:
            test_cfg_override['guidance_interval'] = self.guidance_interval

        # capture the predicted clean latent (x0) at every sampling step
        captured = []

        def sample_callback(net_self, kw):
            x0 = net_self.u_to_x_0(kw['denoising_output'], kw['x_t'], kw['t'])
            captured.append(model.unpatchify(x0).detach())
            return kw

        diffusion(
            noise=noise,
            guidance_scale=self.guidance_scale if use_cfg else 1.0,
            test_cfg_override=test_cfg_override,
            class_labels=class_labels,
            sample_callback=sample_callback)

        if not captured:
            if was_training:
                diffusion.train()
            return

        # subsample to at most max_cols evenly-spaced steps (incl. first & last),
        # so the grid stays readable regardless of nfe / sampler order.
        if self.max_cols and len(captured) > self.max_cols:
            idx = torch.linspace(0, len(captured) - 1, self.max_cols).round().long().tolist()
            captured = [captured[i] for i in idx]

        # decode each step's predicted-clean latent to an image in [0, 1]
        vae = model.vae
        vae_dtype = vae.dtype if hasattr(vae, 'dtype') else next(vae.parameters()).dtype
        imgs = [(vae.decode(lat.to(vae_dtype)).float() / 2 + 0.5).clamp(0, 1)
                for lat in captured]  # each [n, 3, H, W]
        steps = len(imgs)
        # [n, steps, 3, H, W] -> flat sample-major so each grid row is one trajectory
        grid_in = torch.stack(imgs, dim=1).reshape(n * steps, *imgs[0].shape[1:])
        grid = make_grid(grid_in.cpu().float(), nrow=steps)
        grid_np = (grid.permute(1, 2, 0).numpy() * 255).round().astype('uint8')

        wandb.log({self.tag: wandb.Image(grid_np)}, commit=False)

        if was_training:
            diffusion.train()
