"""How many null-bank images does v*_uncond actually use?

x0_hat forms a SELF-NORMALIZED weighted mean, w = softmax(-d^2) over the bank, so
it is biased at any finite M with bias O(1/ESS) -- ESS = 1/sum(w_i^2) being the
effective sample size the softmax leaves after concentration. Raw M only helps
insofar as ESS tracks it. This measures ESS(sigma, M) on real images, and the
convergence of the weighted mean against a large-M reference.
"""
import os, sys, json, time, random
import numpy as np, torch
sys.path.insert(0, '/scratch/gpfs/AM43/zz8976/LakonLab')
os.chdir('/scratch/gpfs/AM43/zz8976/LakonLab')

from mmcv import Config
from mmcv.runner import load_checkpoint
from lakonlab.models import build_model

N_POOL = int(os.environ.get('N_POOL', 65536))
SIGMAS = [0.98, 0.95, 0.92, 0.88, 0.80]
MS     = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
CHUNK  = 2048

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
print(f'pool: drawing {N_POOL} paths from the whole dataset', flush=True)

paths = random.sample(exp._all_paths, min(N_POOL, len(exp._all_paths)))
feats, t0 = [], time.time()
for b in range(0, len(paths), CHUNK):
    u8 = exp._load_images_u8(paths[b:b + CHUNK])
    imgs = u8.permute(0, 3, 1, 2).float() / 255.0
    lat = model._expert_encode_fn(imgs)
    feats.append(d.feat_fn(lat).flatten(1).float())
    print(f'  {b + len(feats[-1]):6d}/{len(paths)}  ({time.time()-t0:.0f}s)', flush=True)
feat_m = torch.cat(feats); del feats
print(f'feats {tuple(feat_m.shape)} in {time.time()-t0:.0f}s', flush=True)

# a realistic band state: a held-out real image noised to sigma
held = exp._load_images_u8(random.sample(exp._all_paths, 1))
x0 = model._expert_encode_fn(held.permute(0, 3, 1, 2).float() / 255.0)
xf0 = d.feat_fn(x0).flatten(1).float()

out = []
for s in SIGMAS:
    xf = (1 - s) * xf0 + s * torch.randn_like(xf0) * 1.0
    resid = xf - (1 - s) * feat_m
    d2 = resid.pow(2).sum(-1) / (2.0 * s ** 2)
    w_full = torch.softmax(-d2, dim=0)
    ref = (w_full.unsqueeze(-1) * feat_m).sum(0)
    for M in MS:
        if M > feat_m.shape[0]:
            continue
        idx = torch.randperm(feat_m.shape[0])[:M]
        wm = torch.softmax(-d2[idx], dim=0)
        ess = float(1.0 / wm.pow(2).sum())
        est = (wm.unsqueeze(-1) * feat_m[idx]).sum(0)
        rel = float((est - ref).norm() / ref.norm().clamp_min(1e-12))
        out.append(dict(sigma=s, M=M, ess=ess, ess_frac=ess / M, rel_err=rel))
    hit = [r for r in out if r['sigma'] == s and r['M'] == 2048]
    if hit:
        print(f"sigma {s}: at M=2048  ESS={hit[0]['ess']:.0f}  "
              f"ESS/M={hit[0]['ess_frac']:.3f}  rel_err={hit[0]['rel_err']:.4f}", flush=True)

json.dump(out, open('direct_outputs/null_bank_ess.json', 'w'), indent=1, default=float)
print(f'\n{"sigma":>6} {"M":>7} {"ESS":>10} {"ESS/M":>8} {"rel err vs full":>16}')
for r in out:
    print(f'{r["sigma"]:6.2f} {r["M"]:7d} {r["ess"]:10.1f} {r["ess_frac"]:8.3f} {r["rel_err"]:16.4f}')
