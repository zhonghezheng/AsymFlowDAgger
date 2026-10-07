"""Does the MMD band rollout keep gradient through EVERY step, end to end?

Static reading says yes (mmd_t_hi=None -> n_pre=0 -> no torch.no_grad() leg), but
the chain also has to survive _ckpt_pred's checkpointing and sampler.step. This
checks it on the real model: that every returned state carries a grad_fn, that the
LAST state still depends on the FIRST (so the graph spans the whole trajectory and
is not silently re-rooted each step), and that gradient reaches the parameters from
the earliest state.
"""
import os, sys, random
import torch
sys.path.insert(0, '/scratch/gpfs/AM43/zz8976/LakonLab'); os.chdir('/scratch/gpfs/AM43/zz8976/LakonLab')
from mmcv import Config
from mmcv.runner import load_checkpoint
from lakonlab.models import build_model

os.environ.setdefault('LAKON_MMD_FEATURE', 'subspace'); os.environ.setdefault('LAKON_MMD_WEIGHT', '10')
os.environ.setdefault('LAKON_MMD_CLASSES', '8');        os.environ.setdefault('LAKON_MMD_SHARE', 'both')
cfg = Config.fromfile('configs/asymflow/asymflow_h_16_r8_imagenet_regft_mmd_sweep_4gpus.py')
model = build_model(cfg.model)
model = model.cuda().train()
d = model.diffusion
B = 4
x0 = torch.randn(B, 3, 256, 256, device='cuda')
lab = torch.randint(0, 1000, (B,), device='cuda')
t_split = d.mmd_t_split

states = d._band_rollout(x0, lab, t_split)
print(f'n_states = {len(states)}   mmd_t_hi = {d.mmd_t_hi}  (n_pre=0 expected)\n')
print(f"{'k':>3} {'sigma':>7} {'requires_grad':>14} {'grad_fn':>28}")
for k, (s, x) in enumerate(states):
    print(f'{k:>3} {s:>7.4f} {str(x.requires_grad):>14} {type(x.grad_fn).__name__ if x.grad_fn else "None":>28}')

first, last = states[0][1], states[-1][1]
g = torch.autograd.grad(last.sum(), first, retain_graph=True, allow_unused=True)[0]
print(f'\n  d(last)/d(first)  -> {"CONNECTED, |g|=%.3e" % g.abs().sum() if g is not None else "NONE (graph broken between steps)"}')

params = [p for p in d.denoising.parameters() if p.requires_grad]
for k in (0, len(states) // 2, len(states) - 1):
    gs = torch.autograd.grad(states[k][1].sum(), params, retain_graph=True, allow_unused=True)
    nz = sum(1 for x in gs if x is not None and x.abs().sum() > 0)
    print(f'  state[{k}] (sigma={states[k][0]:.3f}) -> params with nonzero grad: {nz}/{len(params)}')
print('\nPASS if every state has a grad_fn, last<-first is CONNECTED, and state[0] already reaches params.')
