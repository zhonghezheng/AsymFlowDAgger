"""Which velocity did the RELEASED checkpoint learn: eps - x1, or P.eps - x1?

The two hypotheses differ in the orthogonal complement by exactly eps_comp, a
unit-scale vector at band sigmas, so one forward pass on a real (x1, eps) pair
separates them decisively. Uses the TRUE x1 and eps (no expert, no estimate).
"""
import os, sys, random
import torch
sys.path.insert(0, '/scratch/gpfs/AM43/zz8976/LakonLab')
os.chdir('/scratch/gpfs/AM43/zz8976/LakonLab')
from mmcv import Config
from mmcv.runner import load_checkpoint
from lakonlab.models import build_model

CKPT = os.environ.get('CKPT', 'models/asymflow_h_16_r8_imagenet.safetensors')
SIGMAS = [0.98, 0.95, 0.92, 0.80, 0.50]
N = 8

cfg = Config.fromfile('configs/asymflow/asymflow_h_16_r8_imagenet_dagger_bankfull_cfg_4gpus.py')
model = build_model(cfg.model)
load_checkpoint(model, CKPT, map_location='cpu')
model = model.cuda().eval()
for p in model.parameters(): p.requires_grad_(False)
d = model.diffusion
exp = d._dagger_expert; exp.ready or exp.prepare()

u8 = exp._load_images_u8(random.sample(exp._all_paths, N))
x1 = model._expert_encode_fn(u8.permute(0, 3, 1, 2).float().cuda() / 255.0)   # DATA
print(f'x1 {tuple(x1.shape)}  |x1|/elem={x1.pow(2).mean().sqrt():.4f}', flush=True)

Pc = lambda z: z - d.project_fn(z)          # complement part

print(f"\n{'sigma':>6} {'|comp(model)|':>14} {'|eps_c - x1_c|':>15} {'|-x1_c|':>10}"
      f" {'err vs FULL':>12} {'err vs PROJ':>12}")
for s in SIGMAS:
    sig = torch.full((N, 1, 1, 1), s, device='cuda')
    eps = torch.randn_like(x1)
    x_t = (1 - sig) * x1 + sig * eps
    t = sig.reshape(N) * d.num_timesteps
    with torch.no_grad():
        out = d.pred(x_t, t, class_labels=torch.full((N,), d.null_label, device='cuda'))
    _, _, cc = d.get_clamp_coef(t=t, x_t=x_t)
    v = out * cc                                          # model velocity, clamp-weighted
    vc   = Pc(v)
    full = Pc(eps - x1)                                   # eps_c - x1_c
    proj = Pc(-x1)                                        # -x1_c
    r = lambda a, b: ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()
    print(f"{s:>6} {vc.pow(2).mean().sqrt():>14.4f} {full.pow(2).mean().sqrt():>15.4f}"
          f" {proj.pow(2).mean().sqrt():>10.4f} {r(vc, full):>12.3f} {r(vc, proj):>12.3f}",
          flush=True)
print("\nrelative error vs each hypothesis; the SMALLER column is what the checkpoint learned.")
