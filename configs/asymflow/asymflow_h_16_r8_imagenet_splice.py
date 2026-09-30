"""Inference-time expert-velocity SPLICE test on the BASE checkpoint.

During sampling: for sigma > t_split use the empirical expert's CONDITIONAL
velocity toward the per-class posterior mean (full class bank); for sigma <=
t_split use the model's own predicted velocity. Probes whether the expert's
high-sigma target is *good* (splicing it in helps) or *bad* (it hurts) --
independent of DAGGER training.

Model = LatentDiffusionClassImageDagger (base weights via denoising.pretrained;
no --ckpt). test_cfg.expert_splice=True routes val_step through the splice path
(see latent_diffusion_class_image_dagger.py). bank_size=None -> full class.
Guidance is swept over the LOW-sigma model region only (guidance is off at high
sigma under interval [0,0.88], where the expert now drives).

Single-GPU (each rank caches the full-class banks ~ dataset size in host RAM, so
DDP would multiply it); the first guidance setting loads the dataset once, later
settings reuse the cache.
"""

import os

_base_ = ['./asymflow_h_16_r8_imagenet_dagger_4gpus.py']

name = 'asymflow_h_16_r8_imagenet_splice'
work_dir = f'work_dirs/{name}'

# keep the expert (needed for the splice); full class per conditional bank
model = dict(expert=dict(bank_size=None))

# routes val_step through the expert-splice path
test_cfg = dict(expert_splice=True, t_split=0.92, latent_size=(3, 256, 256))

num_images = int(os.environ.get('LAKON_EVAL_NUM_IMAGES', 10000))
data = dict(val=dict(num_test_images=num_images))

step = 50
# CFG upper cutoff (sigma <= hi is guided); env-overridable so the same config can
# sweep intervals. Default 0.88 (the run default).
guidance_interval = [0, float(os.environ.get('LAKON_CFG_HI', 0.88))]

# conditional + CFG only (uncond generation + conditional splice is incoherent)
# focused around the CFG optimum (base ~2.4-2.6, splice ~2.2-2.4); env-overridable
# so a single value can be added, e.g. LAKON_GUIDANCE=1.8
_gvals = os.environ.get('LAKON_GUIDANCE', '2.0,2.2,2.4,2.6')
guidance_settings = [
    (float(g), 'cond' if float(g) == 1.0 else 'cfg') for g in _gvals.split(',')
]


def _prefix(g, tag):
    if g in (0.0, 1.0):
        return f'{tag}_heun_g{g}_step{step}'
    return f'{tag}_heun_g{g}({guidance_interval[0]}-{guidance_interval[1]})_step{step}'


evaluation = [
    dict(
        type='GenerativeEvalHook',
        data='val',
        prefix=_prefix(g, tag),
        interval=1,
        sample_kwargs=dict(
            test_cfg_override=dict(
                sampler='FlowHeunODE',
                guidance_scale=g,
                guidance_interval=guidance_interval,
                num_timesteps=step,
            ),
        ),
        feed_batch_size=32,
        metrics=[
            dict(
                type='InceptionMetrics',
                num_images=num_images,
                reference_pkl='models/imagenet256_inception_adm.pkl',
                inception_args=dict(
                    type='StyleGAN',
                    inception_path='models/inception-2015-12-05.pt'),
            ),
        ],
        save_best_ckpt=False,
    )
    for g, tag in guidance_settings
]
