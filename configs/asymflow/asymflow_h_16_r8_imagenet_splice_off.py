_base_ = ['./asymflow_h_16_r8_imagenet_splice.py']
name = 'asymflow_h_16_r8_imagenet_splice_off'
work_dir = f'work_dirs/{name}'
test_cfg = dict(expert_splice=False, latent_size=(3, 256, 256))  # plain base-model eval
model = dict(diffusion_use_ema=False)  # use pretrained diffusion, not the (untrained) EMA
