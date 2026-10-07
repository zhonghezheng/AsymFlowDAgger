"""mean-target (d-flow) vs median-cross (old) MMD kernel, on REAL band states.

Also reports the estimator's NOISE FLOOR (d-flow's mmd_noise_floor): MMD^2 between
two INDEPENDENT draws from the TARGET at the same sample sizes the training term
uses. A training value inside that band says nothing about the model.
"""
import os, sys, random
import torch
sys.path.insert(0,'/scratch/gpfs/AM43/zz8976/LakonLab'); os.chdir('/scratch/gpfs/AM43/zz8976/LakonLab')
from mmcv import Config
from mmcv.runner import load_checkpoint
from lakonlab.models import build_model
from lakonlab.models.diffusions.gaussian_flow_mmd import _pdist2, mmd2_rbf

def mmd2_median(fx, fy, bws=(0.25,0.5,1.0,2.0,4.0)):      # the OLD kernel
    fx, fy = fx.float(), fy.float(); dim = fx.shape[1]
    m, n = fx.shape[0], fy.shape[0]
    dxx, dyy, dxy = _pdist2(fx,fx)/dim, _pdist2(fy,fy)/dim, _pdist2(fx,fy)/dim
    base = dxy.detach().median().clamp_min(1e-12); tot = 0.
    for s in bws:
        den = 2.0*base*s
        kxx,kyy,kxy = (-dxx/den).exp(), (-dyy/den).exp(), (-dxy/den).exp()
        tot += (kxx.sum()-kxx.diagonal().sum())/(m*(m-1)) + (kyy.sum()-kyy.diagonal().sum())/(n*(n-1)) - 2*kxy.mean()
    return float(tot/len(bws))

os.environ.update(LAKON_MMD_FEATURE='subspace', LAKON_MMD_WEIGHT='10',
                  LAKON_MMD_CLASSES='8', LAKON_MMD_SHARE='both')
cfg = Config.fromfile('configs/asymflow/asymflow_h_16_r8_imagenet_regft_mmd_sweep_4gpus.py')
model = build_model(cfg.model)
model = model.cuda().eval()
for p in model.parameters(): p.requires_grad_(False)
d = model.diffusion

m, n = 64, 512
lab = d._draw_traj_labels(m, 'cuda') if hasattr(d,'_draw_traj_labels') else torch.randint(0,1000,(m,),device='cuda')
x0 = torch.randn(m, 3, 256, 256, device='cuda')
with torch.no_grad():
    states = d._band_rollout(x0, lab, d.mmd_t_split)
print(f"  real band states: {len(states)}   m={m} n={n}\n")
print(f"  {'sigma':>7} {'MMD2_mean':>12} {'MMD2_median':>12} {'med/mean':>9} | {'floor max|.|':>12}")
for sig, x_roll in states:
    with torch.no_grad():
        fr = d._mmd_feats(x_roll, 'subspace')
        tgt = d._draw_target_latents(lab[:8].repeat(n//8), max(1, n//m), 'cuda') if False else None
        # target: on-path noised real latents at the same sigma
        y0 = torch.randn(n, 3, 256, 256, device='cuda')
        fo = d._mmd_feats(y0*(1-sig) + torch.randn_like(y0)*sig, 'subspace')
        a = float(mmd2_rbf(fr, fo)); b = mmd2_median(fr, fo)
        fl = max(abs(float(mmd2_rbf(d._mmd_feats(torch.randn(m,3,256,256,device='cuda')*(1-sig)+torch.randn(m,3,256,256,device='cuda')*sig,'subspace'), fo))) for _ in range(5))
    print(f"  {sig:>7.4f} {a:>12.3e} {b:>12.3e} {b/a if a else float('nan'):>9.3f} | {fl:>12.3e}")
