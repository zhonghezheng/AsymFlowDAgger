name = 'asymflow_h_16_r8_imagenet_dagger_4gpus'


# --- finetuning schedule (from the released checkpoint) ---
steps_per_epoch = 2502  # 1.28M / (4 gpus * 128)
total_iters = 20000
warmup_iters = 200
save_interval = 2000
must_save_interval = 10000
eval_interval = 5000

# --- DAGGER knobs ---
latent_size = (3, 256, 256)
round_interval = 500     # iters between rollout rounds
n_rollout = 128          # trajectories rolled out per round
rollout_nfe = 20         # Euler steps captured per rollout (buffer ~= n_rollout * nfe points)
# timestep split: on-path FM covers sigma < t_split (data side), DAGGER rollout/
# expert covers sigma >= t_split (noise side). Each train minibatch is carved to a
# fixed batch_size, split by the base logit-normal mass p_high = P(sigma >= t_split):
# n_on = round(bs*(1-p_high)) on-path points, n_roll = bs - n_on rollout points.
t_split = 0.8
complement_mode = 'project'  # 'project' (derived asym loss) | 'full' (keep complement noise)

model = dict(
    type='LatentDiffusionClassImageDagger',
    expert=dict(
        type='EmpiricalExpert',
        # per-class reservoir of real data latents (CPU): num_classes * per_class_pool
        # latents total. At 3x256x256, 1000*64 ~= 50 GB/rank (x4 ranks in RAM).
        num_classes=1000,
        per_class_pool=64,   # guaranteed members per class (on-path uses all of them)
        bank_k=1024,         # sampled bank per rollout trajectory; cap for null on-path
        sample_chunk=32,     # rows per chunk
        bank_chunk=128,      # bank entries summed at once (peak ~ chunk*bank_chunk latents)
    ),
    vae=dict(
        type='RGBColorEncoder',
    ),
    diffusion=dict(
        type='GaussianFlowDagger',
        complement_mode=complement_mode,
        roll_weight=1.0,
        t_split=t_split,
        # label on-path samples with the empirical expert velocity instead of the
        # true FM residual (noise - x_0). Set False to keep standard FM on-path.
        onpath_expert_vel=True,
        denoising=dict(
            type='AsymJiT',
            patch_size=16,
            in_channels=3,
            basis_rank=8,
            num_timesteps=1,
            pretrained_linear_proj='checkpoints/asymflow_subspace_pca_dit.pth',
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
            checkpointing=True,
            sigma_min=4e-2,  # AsymFlow inference clamp
        ),
        flow_loss=dict(
            type='DiffusionMSELoss',
            data_info=dict(pred='u_t_pred', target='u_t'),
            rescale_mode='constant',
            rescale_cfg=dict(scale=2.0),  # LakonLab MSE loss has a internal 0.5 factor, so use 2.0
        ),
        num_timesteps=1,
        timestep_sampler=dict(
            type='ContinuousTimeStepSampler',
            shift=1.0,
            logit_normal_enable=True,
            logit_normal_mean=0.8,
            logit_normal_std=0.8,
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
        lr=5e-5,  # lowered for finetuning
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
        image_size=256),
    train_dataloader=dict(samples_per_gpu=128),
    val=dict(
        type='ImageNet',
        data_root='data/imagenet/train/',
        datalist_path='data/imagenet/train.txt',
        negative_label=1000,
        latent_size=(3, 256, 256),
        test_label_sampling='equal',
        test_mode=True),
    val_dataloader=dict(samples_per_gpu=64),
    test_dataloader=dict(samples_per_gpu=64),
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=32,
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
                num_images=50000,
                reference_pkl='huggingface://Lakonik/inception_feats/imagenet256_inception_adm.pkl',
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
        n_rollout=n_rollout,
        nfe=rollout_nfe,
        latent_size=latent_size,
        num_classes=1000,
        null_label=1000,
        # prob_class omitted -> inherits train_cfg.prob_class (0.9), so the
        # DAGGER CFG dropout stays inline with the on-path stream automatically.
        # t_split omitted -> inherits diffusion.t_split (0.5); captures sigma >= t_split.
        rollout_chunk=64,
        guidance_scale=guidance_scale,        # inference CFG, so visited states match sampling
        guidance_interval=guidance_interval,
        use_ema_rollout=False,
        aggregate_buffer=False,
        priority='NORMAL'),
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
load_from = 'models/asymflow_h_16_r8_imagenet.safetensors'
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
