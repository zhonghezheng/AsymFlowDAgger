"""Is the mmd_chunk path the same objective and gradient as the single-graph path?

Run on >= 2 ranks (torchrun): at world_size 1 the sharding and all-reduces are no-ops.

  A. mmd2_rbf_sharded vs mmd2_rbf on the pooled sets (TF32 off for both): the value,
     and the gradient w.r.t. this rank's rows (the reference differentiates the pooled
     loss through the local slice of a gathered tensor, as _gather_feats does).
  B. End to end on the real model at small size: _mmd_chunked_step's parameter
     gradient, in both modes (single pass: graph kept, scored on detached copies;
     replay: scored without a graph, replayed in chunks), vs ONE graph over the same
     trajectories -- same noise, labels and fixed targets -- scored with mmd2_rbf and
     differentiated directly. All are DDP-averaged (all_reduce / W) before comparing.

PASS: A rel errors ~1e-6 (fp32 rounding); B rel error well below 1e-2 and cosine ~1 (the
two paths run the same bf16 network evals, with and without a graph).
"""
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, '/scratch/gpfs/AM43/zz8976/LakonLab')
os.chdir('/scratch/gpfs/AM43/zz8976/LakonLab')
from mmcv import Config  # noqa: E402

from lakonlab.models import build_model  # noqa: E402
from lakonlab.models.diffusions.gaussian_flow_mmd import (  # noqa: E402
    _no_tf32, mmd2_rbf, mmd2_rbf_sharded)

dist.init_process_group('nccl')
rank, ws = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', rank)))
dev = torch.device('cuda')


def say(*a):
    if rank == 0:
        print(*a, flush=True)


def gather_local_graph(f):
    buf = [torch.empty_like(f) for _ in range(ws)]
    dist.all_gather(buf, f.detach().contiguous())
    buf[rank] = f
    return torch.cat(buf, dim=0)


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


failures = []


def check(ok, what):
    if not ok:
        failures.append(what)


# ---------------------------------------------------------------- A. estimator
say(f'== A. mmd2_rbf_sharded vs mmd2_rbf (world_size={ws})')
torch.manual_seed(1234 + rank)
D = 3 * 256 * 256
for sigma, (m_loc, n_loc) in [(0.88, (64, 320)), (0.98, (32, 160))]:
    data_x = torch.randn(m_loc, D, device=dev) * 1.1   # slightly too wide model
    data_y = torch.randn(n_loc, D, device=dev)
    fx = ((1 - sigma) * data_x + sigma * torch.randn(m_loc, D, device=dev)).requires_grad_(True)
    fy = (1 - sigma) * data_y + sigma * torch.randn(n_loc, D, device=dev)

    with _no_tf32():
        ref = mmd2_rbf(gather_local_graph(fx), gather_local_graph(fy))
    g_ref = torch.autograd.grad(ref, fx)[0]
    val, sur = mmd2_rbf_sharded(fx, fy)
    g = torch.autograd.grad(sur, fx)[0]
    say(f'  sigma={sigma} m={m_loc * ws} n={n_loc * ws}: ref={float(ref):+.4e} '
        f'sharded={float(val):+.4e}  |dval|={abs(float(val - ref)):.2e}  '
        f'grad rel err={rel(g, g_ref):.2e}')
    check(abs(float(val - ref)) < 1e-5 and rel(g, g_ref) < 1e-3, f'A sigma={sigma}')

# ---------------------------------------------------------------- B. end to end
say('\n== B. chunked step vs one graph, real model')
cfg = Config.fromfile('configs/asymflow/asymflow_h_16_r8_imagenet_regft_mmd_m2048_4gpus.py')
cfg.model.diffusion.denoising.compile_forward = False   # same eager evals on both sides
# Eager per-block checkpoints recompute in whatever mode the net is in at BACKWARD time:
# _ckpt_pred rolls out in eval mode but restores train() before the backward, so the
# recompute would run proj_dropout and trip CheckpointError (saved-tensor metadata
# mismatch). The compiled training path is immune -- the recompute is fixed at trace
# time. At m=8 the activations are small, so just store them.
cfg.model.diffusion.denoising.checkpointing = False
model = build_model(cfg.model).to(dev).train()
d = model.diffusion
m, chunk, n_loc = 8, 4, 16   # per rank: 8 rollouts (replay: 2 chunks of 4), 16 targets
d.mmd_batch = m
d.mmd_target_share = 'none'
t_split = d.mmd_t_split

torch.manual_seed(99 + rank)
labels = torch.randint(0, 1000, (m, ), device=dev)
noise = torch.randn(m, 3, 256, 256, device=dev)
x_0 = torch.randn(m, 3, 256, 256, device=dev)
tgt_x0 = torch.randn(n_loc, 3, 256, 256, device=dev).clamp(-1, 1)
tgt_eps = torch.randn_like(tgt_x0)
# fixed targets on both sides: one straight-line set, as mmd_target_share='both' gives
d._target_latents = lambda x0_, sigma, *a, **k: (1 - sigma) * tgt_x0 + sigma * tgt_eps
params = [p for p in d.denoising.parameters() if p.requires_grad]

# reference: one graph over all m trajectories
states = d._band_rollout(noise, labels, t_split, noise=noise)
n_steps = len(states)
loss_ref, vals_ref = 0.0, []
for sigma, x in states:
    with _no_tf32():
        v = mmd2_rbf(gather_local_graph(x.flatten(1).float()),
                     gather_local_graph(d._target_latents(None, sigma).flatten(1)),
                     bandwidths=d.mmd_bandwidths)
    vals_ref.append(float(v))
    loss_ref = loss_ref + v
loss_ref = loss_ref * d.mmd_weight * d.mmd_accum_steps * ws / n_steps
g_ref = torch.autograd.grad(loss_ref, params, allow_unused=True)
g_ref = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                   for g, p in zip(g_ref, params)])
del states, loss_ref

dist.all_reduce(g_ref)
g_ref /= ws
acc = float(d.mmd_accum_steps)
say(f'  per-step MMD^2 ref : {" ".join(f"{v:+.3e}" for v in vals_ref)}  |g|={float(g_ref.norm()):.4e}')

# both modes of the mmd_chunk path against the one-graph reference
for mode, ck in (('single pass', m), ('replay', chunk)):
    d.mmd_chunk = ck
    for p in params:
        p.grad = None
    log_vars = d._mmd_chunked_step(x_0, labels, t_split, noise=noise)
    g_chk = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                       for p in params])
    dist.all_reduce(g_chk)
    g_chk /= ws
    vals_chk = [float(v) / acc for k, v in log_vars.items() if k.startswith('mmd_raw_s')]
    cos = float(torch.nn.functional.cosine_similarity(g_chk, g_ref, dim=0))
    say(f'  [{mode}, chunk={ck}]')
    say(f'    per-step MMD^2   : {" ".join(f"{v:+.3e}" for v in vals_chk)}')
    say(f'    |g|={float(g_chk.norm()):.4e}  grad rel err = {rel(g_chk, g_ref):.3e}  '
        f'cosine = {cos:.6f}')
    if 'mmd_replay_err' in log_vars:
        say(f'    replay_err = {float(log_vars["mmd_replay_err"]) / acc:.3e}')
    check(rel(g_chk, g_ref) < 2e-2 and cos > 0.999, f'B gradient ({mode})')
    check(all(abs(a - b) < 1e-5 + 1e-2 * abs(b) for a, b in zip(vals_chk, vals_ref)),
          f'B values ({mode})')
say('\nRESULT:', 'PASS' if not failures else f'FAIL {failures}')
dist.barrier()
dist.destroy_process_group()
sys.exit(1 if failures else 0)
