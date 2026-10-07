"""What does temp_spread actually buy? ESS, variance and BIAS of x0_hat.

temp_spread is not variance reduction of a fixed estimand: dividing d2 by T moves
the target from the Gaussian posterior mean (T=1, correct but ESS~1 so enormous
finite-M variance) toward the plain bank MEAN (T->inf, zero variance but
uninformative -- the mean-reverting high-sigma failure the DAGGER arms already hit).
So the useful number is the bias/variance split against the T=1 full-pool reference,
not ESS on its own.

For each (sigma, temp_spread): draw R independent banks of M images, form x0_hat in
FEATURE space exactly as EmpiricalExpert.x0_hat does, and report
  ess   -- mean effective sample size of the softmax
  sd    -- across-draw std of x0_hat (the variance the target injects)
  bias  -- |mean(x0_hat) - reference| , reference = T=1 over the whole pool
  rmse  -- sqrt(bias^2 + sd^2), the quantity a regression target actually pays
all relative to the reference's own norm.
"""
import os, sys, json, time, random
import torch
sys.path.insert(0, '/scratch/gpfs/AM43/zz8976/LakonLab')
os.chdir('/scratch/gpfs/AM43/zz8976/LakonLab')

from mmcv import Config
from mmcv.runner import load_checkpoint
from lakonlab.models import build_model

N_POOL  = int(os.environ.get('N_POOL', 32768))
M       = int(os.environ.get('M', 2048))
R       = int(os.environ.get('R', 24))          # independent bank draws
SIGMAS  = [0.92]
MS      = [512, 2048, 8192]   # does bank size matter once temperature is on?
SPREADS = [None, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 20.0]  # optimum was past 3.0
CHUNK   = 2048

cfg = Config.fromfile('configs/asymflow/asymflow_h_16_r8_imagenet_dagger_bankfull_cfg_4gpus.py')
model = build_model(cfg.model)
load_checkpoint(model, 'checkpoints/asymflow_h_16_r8_imagenet_regft_4gpus/latest.pth',
                map_location='cpu')
model = model.eval()
for p in model.parameters():
    p.requires_grad_(False)
d = model.diffusion
exp = d._dagger_expert
exp.ready or exp.prepare()

paths = random.sample(exp._all_paths, min(N_POOL, len(exp._all_paths)))
feats, t0 = [], time.time()
for b in range(0, len(paths), CHUNK):
    u8 = exp._load_images_u8(paths[b:b + CHUNK])
    lat = model._expert_encode_fn(u8.permute(0, 3, 1, 2).float() / 255.0)
    feats.append(d.feat_fn(lat).flatten(1).float())
    print(f'  {b + feats[-1].shape[0]:6d}/{len(paths)} ({time.time()-t0:.0f}s)', flush=True)
feat_m = torch.cat(feats); del feats
print(f'pool {tuple(feat_m.shape)} in {time.time()-t0:.0f}s', flush=True)

held = exp._load_images_u8(random.sample(exp._all_paths, 1))
xf0 = d.feat_fn(model._expert_encode_fn(held.permute(0, 3, 1, 2).float() / 255.0)).flatten(1).float()

def x0_hat(xf, bank, spread):
    """Feature-space x0_hat with the production temperature rule."""
    d2 = (xf - (1 - S) * bank).pow(2).sum(-1) / (2.0 * S ** 2)
    if spread is not None and d2.numel() > 1:
        d2 = d2 / (d2.std() / spread).clamp_min(1.0)
    w = torch.softmax(-d2, dim=0)
    return (w.unsqueeze(-1) * bank).sum(0), float(1.0 / w.pow(2).sum())

out = []
for S in SIGMAS:
    xf = (1 - S) * xf0 + S * torch.randn_like(xf0)
    for sp in SPREADS:
      for M in MS:
        # reference at the SAME spread over the WHOLE pool. The earlier version used
        # T=1, whose ESS is ~1 -- a single nearest image -- so its 'bias' was mostly
        # the gap between two arbitrary neighbours, not finite-bank error. Matching
        # the spread isolates exactly the question being asked: how much does using
        # M rows instead of the full pool cost, at this temperature.
        ref, _ = x0_hat(xf, feat_m, sp)
        rn = ref.norm().clamp_min(1e-12)
        ests, esss = [], []
        for _ in range(R):
            idx = torch.randperm(feat_m.shape[0])[:M]
            e, ess = x0_hat(xf, feat_m[idx], sp)
            ests.append(e); esss.append(ess)
        E = torch.stack(ests)
        bias = (E.mean(0) - ref).norm() / rn
        sd = E.std(0).norm() / rn
        ess = sum(esss) / len(esss)
        rec = dict(sigma=S, spread=sp, M=M, ess=ess, ess_frac=ess / M,
                   bias=float(bias), sd=float(sd),
                   rmse=float((bias ** 2 + sd ** 2).sqrt()))
        out.append(rec)
        print(f"  spread={str(sp):5s} M={M:<6} ESS={ess:8.1f} ESS/M={ess/M:.4f}  "
              f"bias={rec['bias']:.4f} sd={rec['sd']:.4f} rmse={rec['rmse']:.4f}", flush=True)

json.dump(out, open('direct_outputs/temp_spread_ess.json', 'w'), indent=1, default=float)
print('\nwrote direct_outputs/temp_spread_ess.json')
