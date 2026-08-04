"""Pearson vs copula SpatialDHGNN incidence must differ on identical IV history."""
from __future__ import annotations

import torch

from src.features.dhgnn import SpatialDHGNN, _pearson_tail_incidence
from src.features.extractor import AlphaFeatureExtractor


def test_pearson_and_copula_incidence_differ() -> None:
    torch.manual_seed(0)
    k, t, d = 4, 32, 8
    # Correlated IV levels so Pearson |corr| of log-changes is structured;
    # copula ranks of levels produce a different dependence geometry.
    base = torch.randn(t, 1).abs() + 0.15
    noise = torch.randn(t, k) * 0.02
    iv_hist = (base + noise).clamp_min(0.05)

    pearson = SpatialDHGNN(
        d_model=d,
        num_assets=k,
        spatial_mode="dhgnn_pearson",
        hist_len=t,
        allow_pearson_incidence=True,
    )
    copula = SpatialDHGNN(
        d_model=d, num_assets=k, spatial_mode="dhgnn_copula", hist_len=t
    )
    # Seed identical history into both (bypass EMA training path).
    pearson.iv_hist[:t] = iv_hist
    pearson.iv_hist_count.fill_(t)
    pearson.running_tail.copy_(
        _pearson_tail_incidence(iv_hist, threshold=pearson.edge_threshold)
    )
    copula.iv_hist[:t] = iv_hist
    copula.iv_hist_count.fill_(t)
    copula.running_tail.copy_(
        copula._empirical_copula_tail_dependence(iv_hist)
    )

    iv_batch = iv_hist[-1:].expand(2, -1)  # (B=2, K)
    H_p = pearson._build_dynamic_incidence_matrix(iv_batch)
    H_c = copula._build_dynamic_incidence_matrix(iv_batch)
    assert H_p.shape == H_c.shape == (2, k, k)
    assert not torch.allclose(H_p, H_c), "pearson and copula incidence must differ"


def test_spatial_mode_wired_through_extractor() -> None:
    fe = AlphaFeatureExtractor(
        num_assets=3,
        d_model=8,
        d_state=4,
        temporal_backend="mlp",
        use_dhgnn=True,
        spatial_mode="dhgnn_pearson",
    )
    assert fe.spatial_mode == "dhgnn_pearson"
    assert fe.spatial_dhgnn.spatial_mode == "dhgnn_pearson"


def test_update_incidence_at_eval_pushes_history() -> None:
    torch.manual_seed(1)
    k, d = 4, 8
    gnn = SpatialDHGNN(
        d_model=d,
        num_assets=k,
        hist_len=16,
        update_incidence_at_eval=True,
    )
    gnn.eval()
    assert not gnn.training
    before = int(gnn.iv_hist_count.item())
    iv = torch.rand(2, k) * 0.2 + 0.1
    _ = gnn._tail_matrix(iv)
    assert int(gnn.iv_hist_count.item()) == before + 1


def test_default_eval_does_not_push_history() -> None:
    torch.manual_seed(2)
    k, d = 4, 8
    gnn = SpatialDHGNN(d_model=d, num_assets=k, hist_len=16)
    gnn.eval()
    before = int(gnn.iv_hist_count.item())
    iv = torch.rand(2, k) * 0.2 + 0.1
    _ = gnn._tail_matrix(iv)
    assert int(gnn.iv_hist_count.item()) == before


def test_training_batches_do_not_push_history() -> None:
    torch.manual_seed(3)
    k, d = 4, 8
    gnn = SpatialDHGNN(d_model=d, num_assets=k, hist_len=16)
    gnn.train()
    before = int(gnn.iv_hist_count.item())
    iv_a = torch.rand(2, k) * 0.2 + 0.1
    iv_b = iv_a.flip(dims=(0,))
    _ = gnn._tail_matrix(iv_a)
    _ = gnn._tail_matrix(iv_b)
    assert int(gnn.iv_hist_count.item()) == before


def test_observe_iv_step_increments_history() -> None:
    k, d = 4, 8
    gnn = SpatialDHGNN(d_model=d, num_assets=k, hist_len=16)
    before = int(gnn.iv_hist_count.item())
    gnn.observe_iv_step(torch.rand(k) * 0.2 + 0.1)
    assert int(gnn.iv_hist_count.item()) == before + 1


def test_spatial_mode_none_skips_dhgnn_attachment() -> None:
    from src.plugins.registry import build_feature_extractor

    cfg = {
        "spatial_mode": "none",
        "use_dhgnn": False,
        "temporal_backend": "mlp",
        "architecture": "mlp",
    }
    fe = build_feature_extractor(4, 16, d_state=8, cfg=cfg, plugins={})
    assert fe.use_dhgnn is False
    assert fe.spatial_dhgnn is None
