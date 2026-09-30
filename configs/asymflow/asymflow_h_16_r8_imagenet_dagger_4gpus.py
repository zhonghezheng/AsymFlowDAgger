name = 'asymflow_h_16_r8_imagenet_dagger_4gpus'


# --- finetuning schedule (from the released checkpoint) ---
steps_per_epoch = 1251  # 1.28M / (4 gpus * 256)
# short comparison run: 10 DAGGER rounds (round_interval=500 -> 10 * 500 = 5000 iters),
# enough to get signal on dagger vs regular finetuning.
total_iters = 5000
warmup_iters = 500   # optimizer/LR warmup on pure on-path FM; DAGGER rounds start after this
save_interval = 2500
must_save_interval = 5000
eval_interval = 500     # eval (10k FID) + trajectory viz every 500 iters (10 evals over 5000)

# --- DAGGER knobs ---
latent_size = (3, 256, 256)
round_interval = 500     # iters between rollout rounds (10 rounds over 5000 iters)
n_rollout = 1024         # trajectories/round per rank (~2.9x buffer reuse; fits 600G at K=128)
rollout_nfe = 50         # Heun steps per rollout, matching the eval sampler (FlowHeunODE, 50)
# --- timestep mass (shared by DAGGER and regft) ---
# The released checkpoint was PRETRAINED with the logit-normal applied as the
# timestep SAMPLING distribution. Here the same mass is applied as an explicit
# per-point LOSS WEIGHT over uniformly-sampled timesteps instead (sampler
# logit_normal_enable=False + flow_loss rescale_mode='logit_normal'). Identical in
# expectation to the pretraining objective -- E[w] = 1, so loss scale and LR carry
# over -- but it decouples two things the sampled version conflated:
#   1. sigma COVERAGE: every batch now spans the full range, so the high-sigma
#      DAGGER region gets p_high = 1 - t_split = 0.12 of the points (vs 0.068 when
#      the logit-normal did the sampling) -> ~2x the rollout rows, less variance.
#   2. sigma WEIGHTING: restored to the pretraining logit-normal via w(raw_t), so
#      the high-sigma points are down-weighted by 0.068/0.12 = 0.567 and the total
#      mass per region matches pretraining exactly.
# Net effect: more samples where DAGGER acts, same objective as the base model.
flow_shift = 1.0             # sampler warp; must match flow_loss rescale_cfg.shift
logit_normal_mean = 0.8      # pretraining values (asymflow_h_16_r8_imagenet_8gpus.py)
logit_normal_std = 0.8
# timestep split: on-path FM covers sigma < t_split (data side), DAGGER rollout/
# expert covers sigma >= t_split (noise side). Each train minibatch is carved to a
# fixed batch_size, split by the base timestep mass p_high = P(sigma >= t_split),
# MC-estimated from timestep_sampler itself -- UNIFORM now, so p_high = 1 - t_split
# = 0.12: n_on = round(bs*(1-p_high)) on-path points, n_roll = bs - n_on rollout
# points. The logit-normal loss weight then restores the correct mass per region.
# 0.88 aligns the expert coverage (sigma >= 0.88) with the eval CFG interval's
# no-guidance tail (guidance_interval=[0,0.88] -> CFG off for sigma > 0.88), so
# DAGGER corrects exactly the high-noise region that inference leaves unguided.
t_split = 0.92
complement_mode = 'full'  # PINNED: 'full' is the ASSEMBLED form of the derived
# asym target (AsymFlow Eq. 3 + Eq. 5); the loss compares assembled velocities, so
# 'project' would put a raw-head-space target against an assembled prediction.

model = dict(
    type='LatentDiffusionClassImageDagger',
    expert=dict(
        type='EmpiricalExpert',
        # bank drawn fresh from the FULL dataset on disk (no RAM reservoir): loads
        # bank_size real images per class per rollout chunk, encoded with the exact
        # train pipeline. Amortized disk IO instead of a ~TB RAM reservoir, and the
        # bank covers the whole class rather than a sliding window.
        datalist_path='data/imagenet/train.txt',
        data_root='data/imagenet/train/',
        image_size=256,
        num_classes=1000,
        bank_size=128,       # real images per class (x2 with include_flips)
        null_bank_size=256,  # per-trajectory null bank (x2 flips=512); each traj draws its own
        include_flips=True,  # bank holds BOTH h-orientations of every image (2x support, bf16 stored)
        num_workers=32,      # parallel image load/decode threads
        sample_chunk=32,     # rows per chunk in x0_hat
        bank_chunk=128,      # bank entries summed at once (peak ~ chunk*bank_chunk latents)
    ),
    vae=dict(
        type='RGBColorEncoder',
    ),
    diffusion=dict(
        type='GaussianFlowDagger',
        complement_mode=complement_mode,
        # convex loss mix: (1-w)*mean_onpath + w*mean_rollout. 'proportional' sets
        # w = n_roll/bs from the POINT COUNTS, which keeps the mix exact under the
        # logit-normal loss weight too: each point ends up contributing wt_i/bs
        # (wt_i = the rescale weight), i.e. one weighted mean over the whole batch.
        # With uniform sampling, t_split=0.88, bs=256, frac_on_path=0.5:
        # n_on=225, n_high=31, n_high_on=16, n_roll=15 -> w = 15/256 ~= 0.059 of the
        # POINTS, carrying ~0.059 * 0.567 ~= 0.033 of the MASS after reweighting.
        roll_weight='proportional',
        t_split=t_split,
        # blend real data (on-path FM) into the high-sigma rollout region: fraction of
        # the sigma>=t_split budget that is real-data FM (true velocity) vs expert
        # rollout. 0.5 -> half the high-sigma points are real data, half expert.
        frac_on_path=0.5,
        # label on-path samples with the empirical expert velocity instead of the
        # true FM residual (noise - x_0). Set False to keep standard FM on-path.
        onpath_expert_vel=False,
        denoising=dict(
            type='AsymJiT',
            patch_size=16,
            in_channels=3,
            basis_rank=8,
            num_timesteps=1,
            # finetune from the released checkpoint: the AsymJiT weights (incl.
            # proj_buffer / scale_buffer) are loaded here via `pretrained`. This is
            # mutually exclusive with `pretrained_linear_proj` (which is only for
            # from-scratch training) -- the checkpoint already carries the subspace.
            pretrained='models/asymflow_h_16_r8_imagenet.safetensors',
            input_size=256,
            hidden_size=1280,
            depth=32,
            num_heads=16,
            bottleneck_dim=256,
            in_context_len=32,
            in_context_start=10,
            num_classes=1000,
            attn_dropout=0.0,
            proj_dropout=0.2,
            torch_dtype='float32',
            autocast_dtype='bfloat16',
            upcast_attention=True,
            fused_attention=True,
            compile_forward=True,
            # mode='default' = inductor WITHOUT cuda graphs. The default
            # 'reduce-overhead' captures a cudagraph pool per input shape, and the
            # DAGGER pipeline feeds many shapes (on-path carve, buffer, rollout
            # forward_test, eval) -> the pools balloon to 100+ GB and OOM the GPU at
            # bs=256. 'default' keeps inductor's fusion/memory savings, no pools.
            compile_kwargs=dict(mode='default', fullgraph=True, dynamic=False),
            checkpointing=True,
            sigma_min=4e-2,  # AsymFlow inference clamp
        ),
        flow_loss=dict(
            type='DiffusionMSELoss',
            data_info=dict(pred='u_t_pred', target='u_t'),
            # logit-normal mass as a per-point WEIGHT over uniform timesteps (see the
            # timestep-mass note at the top). Applied inside flow_loss, so it covers
            # every stream that routes through it: regft's plain FM loss, DAGGER's
            # on-path loss (both the sigma<t_split rows and the frac_on_path
            # high-sigma mix-in), and DAGGER's expert-labelled rollout term.
            rescale_mode='logit_normal',
            rescale_cfg=dict(
                scale=2.0,  # LakonLab MSE loss has a internal 0.5 factor, so use 2.0
                mean=logit_normal_mean,
                std=logit_normal_std,
                num_timesteps=1,  # must match diffusion.num_timesteps below
                shift=flow_shift,
            ),
        ),
        num_timesteps=1,
        timestep_sampler=dict(
            type='ContinuousTimeStepSampler',
            shift=flow_shift,
            # UNIFORM sampling: sigma ~ U(0, 1) (shift=1.0 -> warp_t is the identity).
            # The logit-normal mass is NOT dropped -- it moves to flow_loss as an
            # explicit per-point weight (rescale_mode='logit_normal'). Must stay False
            # here or the mass is applied twice; GaussianFlow.__init__ asserts this.
            logit_normal_enable=False,
        ),
        denoising_mean_mode='U',
        sigma_min=5e-2,  # training loss weight clamp (same as JiT's official 5e-2)
    ),
    diffusion_use_ema=True,
)

work_dir = f'work_dirs/{name}'
train_cfg = dict(
    prob_class=0.9,
    log_interval=10,
)
test_cfg = dict()

optimizer = {
    'diffusion': dict(
        type='AdamW',
        lr=2.5e-4,  # slightly above the base model's training LR (2e-4, at 8 gpus) so the model actively moves
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    ),
}

data = dict(
    workers_per_gpu=8,
    train=dict(
        type='ImageNet',
        data_root='data/imagenet/train/',
        datalist_path='data/imagenet/train.txt',
        negative_label=1000,
        random_flip=True,   # h-flip aug (default); bank construction matches this
        image_size=256),
    train_dataloader=dict(samples_per_gpu=256),
    val=dict(
        type='ImageNet',
        data_root='data/imagenet/train/',
        datalist_path='data/imagenet/train.txt',
        negative_label=1000,
        latent_size=(3, 256, 256),
        test_label_sampling='equal',
        num_test_images=10000,   # eval generates 10k images (== dataset length) for the FID
        test_mode=True),
    val_dataloader=dict(samples_per_gpu=64),
    test_dataloader=dict(samples_per_gpu=64),
    pin_memory=True,
    persistent_workers=True,
    # 32 was overkill (data_time ~0.004s) and buffered ~100 GB of decoded images
    # across 4 ranks -> host OOM. 4 is plenty to keep the H200s fed.
    prefetch_factor=4,
    multiprocessing_context='fork',
)

lr_config = dict(
    policy='fixed',
    warmup='linear',
    warmup_iters=warmup_iters,
    warmup_ratio=1e-8,
    by_epoch=False,
)

checkpoint_config = dict(
    interval=save_interval,
    must_save_interval=must_save_interval,
    by_epoch=False,
    max_keep_ckpts=1,
    out_dir=f'checkpoints/',
)

step = 50
guidance_scale = 2.3
guidance_interval = [0, 0.88]

prefix = f'heun_g{guidance_scale}({guidance_interval[0]}-{guidance_interval[1]})_step{step}'

evaluation = [
    dict(
        type='GenerativeEvalHook',
        data='val',
        prefix=prefix,
        interval=eval_interval,
        sample_kwargs=dict(
            test_cfg_override=dict(
                sampler='FlowHeunODE',
                guidance_scale=guidance_scale,
                guidance_interval=guidance_interval,
                num_timesteps=step,
            ),
        ),
        feed_batch_size=32,
        metrics=[
            dict(
                type='InceptionMetrics',
                num_images=10000,
                reference_pkl='models/imagenet256_inception_adm.pkl',
                inception_args=dict(
                    type='StyleGAN',
                    inception_path='models/inception-2015-12-05.pt'),
            ),
        ],
        save_best_ckpt=False,
    )
]

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        # wandb removed: the shared scratch fileset is full -> avoid the offline-run
        # writes. FID / losses still go to the text log (slurm .out) + tf events.
    ])

custom_hooks = [
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('diffusion_ema', ),
        interp_mode='lerp',
        interval=1,
        start_iter=0,
        momentum_policy='fixed',
        interp_cfg=dict(momentum=0.9999),
        priority='VERY_HIGH'),
    dict(
        type='DaggerRolloutHook',
        round_interval=round_interval,
        start_iter=warmup_iters,   # no DAGGER rounds until the optimizer warmup completes
        n_rollout=n_rollout,
        nfe=rollout_nfe,
        latent_size=latent_size,
        num_classes=1000,
        null_label=1000,
        # prob_class omitted -> inherits train_cfg.prob_class (0.9), so the
        # DAGGER CFG dropout stays inline with the on-path stream automatically.
        # t_split omitted -> inherits diffusion.t_split (0.88); captures sigma >= t_split.
        # smaller chunk -> fewer per-class banks held at once (all 4 ranks build
        # banks simultaneously); keeps host RAM modest with entire-class banks.
        # 8 halves the concurrent-bank peak (~35 GB across ranks) to fit --mem=300G.
        rollout_chunk=8,
        guidance_scale=1.0,   # unguided: conditional-no-CFG or unconditional (guarded in the hook)
        label_time_dropout=True,         # always-conditional rollout; CFG dropout per captured point
        class_sampling='proportional',   # rollout classes ~ dataset prior (like on-path)
        sampler='FlowHeunODE',           # match the eval sampler so captured states are on-policy
        use_ema_rollout=False,
        aggregate_buffer=False,
        priority='NORMAL'),
    # WandbTrajectoryHook removed: no wandb, and it generated extra eval-time samples.
]

runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,
    pass_training_status=True,
    ckpt_trainable_only=True,
    ckpt_fp16=True,
    ckpt_fp16_ema=True,
    ckpt_bf16_optim=True,
    gc_interval=1000)
dist_params = dict(backend='nccl')
log_level = 'INFO'
# weights come from denoising.pretrained (the released AsymJiT checkpoint has bare
# denoising keys, so it must load there, not via top-level load_from).
load_from = None
resume_from = f'checkpoints/{name}/latest.pth'
workflow = [('train', save_interval)]
module_wrapper = 'ddp'
ddp_kwargs = dict(
    # static_graph must be False: the DAGGER buffer branch is absent until the
    # first rollout round fills the buffer, then adds a second forward -- the
    # graph topology is not identical across iterations.
    find_unused_parameters=False,
    static_graph=False,
    gradient_as_bucket_view=True)
cudnn_benchmark = True
mp_start_method = 'fork'
