"""Tests for the DAGGER empirical-expert finetuning components.

Part A (torch only) validates the asym DAGGER target algebra: on-path, the
projected target
    L = (P x_t - (1 - sigma) P x0 - sigma x0) / sigma
must reduce exactly to  L_asym = P eps - x0  (subspace -> eps_sub - x0_sub,
complement -> -x0_comp). This is the formula AsymFlowVRDagger.expert_target
implements.

Part B (needs the full lakonlab / mmcv env) checks that EmpiricalExpert's
posterior mean concentrates on the right data as sigma shrinks.

Run:  python tests/test_dagger_expert.py
"""
import torch


def _random_projector(patch_dim, basis_rank, seed=0):
    g = torch.Generator().manual_seed(seed)
    B = torch.linalg.qr(torch.randn(patch_dim, basis_rank, generator=g))[0]  # orthonormal cols
    P = B @ B.T
    return P, B


def test_target_reduces_on_path():
    torch.manual_seed(0)
    n, patch_dim, basis_rank = 16, 32, 8
    P, _ = _random_projector(patch_dim, basis_rank)

    x0 = torch.randn(n, patch_dim)          # data
    eps = torch.randn(n, patch_dim)         # noise
    sigma = torch.rand(n, 1) * 0.9 + 0.05   # in [0.05, 0.95]
    x_t = (1 - sigma) * x0 + sigma * eps    # on-path interpolant

    Px_t = x_t @ P
    Px0 = x0 @ P
    L = (Px_t - (1 - sigma) * Px0 - sigma * x0) / sigma          # expert_target, project mode

    # exact on-path reduction
    L_asym = eps @ P - x0
    assert torch.allclose(L, L_asym, atol=1e-5), (L - L_asym).abs().max()

    # subspace: eps_sub - x0_sub ; complement: -x0_comp
    assert torch.allclose(L @ P, (eps - x0) @ P, atol=1e-5)
    comp = L - L @ P
    assert torch.allclose(comp, -(x0 - x0 @ P), atol=1e-5)

    # 'full' mode target is the plain velocity toward the data
    L_full = (x_t - x0) / sigma
    assert torch.allclose(L_full, eps - x0, atol=1e-5)
    print('[ok] test_target_reduces_on_path')


def test_empirical_expert_posterior():
    try:
        from lakonlab.models.diffusions.experts import EmpiricalExpert
    except Exception as e:  # mmcv/lakonlab not importable in this env
        print(f'[skip] test_empirical_expert_posterior ({type(e).__name__}: {e})')
        return

    torch.manual_seed(0)
    C, H, W = 2, 2, 2
    n_classes, K = 8, 32
    feat_fn = lambda z: z.flatten(1)  # trivial "projection" for the test

    # per-class reservoir: push K distinct latents for each class
    latents = torch.randn(n_classes * K, C, H, W)
    labels = torch.arange(n_classes).repeat_interleave(K)  # class c -> rows [cK, cK+K)

    expert = EmpiricalExpert(num_classes=n_classes, per_class_pool=K, bank_k=64,
                             sample_chunk=16, bank_chunk=16)
    expert.push_pool(latents, feat_fn, labels)
    assert expert.ready

    c, s = 3, 5
    sigma = 0.05
    target = latents[c * K + s]                         # a class-c data point
    x_t = (1 - sigma) * target[None] + sigma * torch.randn(1, C, H, W)

    # sampled bank must be class-restricted (global index // K == class)
    bank_idx = expert.sample_bank_idx(torch.tensor([c]), device='cpu')
    assert (bank_idx.flatten() // K == c).all()

    x0_hat = expert.x0_hat(x_t, sigma, feat_fn, bank_idx=bank_idx)
    assert (int((latents - x0_hat).flatten(1).norm(dim=1).argmin()) // K) == c

    # full class-restricted posterior (on-path path) should also stay in-class
    x0_full = expert.x0_hat_full(x_t, sigma, feat_fn, torch.tensor([c]))
    assert (int((latents - x0_full).flatten(1).norm(dim=1).argmin()) // K) == c
    print('[ok] test_empirical_expert_posterior')


if __name__ == '__main__':
    test_target_reduces_on_path()
    test_empirical_expert_posterior()
