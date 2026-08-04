"""RC6_HEADS: entmax-1.5, Tsallis entropy, and registry allowlists."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.conftest import FLOAT_TOL


def test_entmax_alpha1_matches_softmax():
    from src.policy.entmax import entmax

    z = torch.tensor([[1.0, 0.0, -1.0, 0.5]])
    got = entmax(z, alpha=1.0)
    exp = torch.softmax(z, dim=-1)
    assert torch.allclose(got, exp, atol=1e-5)


def test_entmax_alpha2_matches_sparsemax():
    from src.policy.entmax import entmax
    from src.policy.sparsemax import sparsemax

    z = torch.tensor([[1.0, 0.5, -2.0, -3.0]])
    got = entmax(z, alpha=2.0)
    exp = sparsemax(z)
    assert torch.allclose(got, exp, atol=1e-5)


def test_entmax_15_between_softmax_and_sparsemax_support():
    from src.policy.entmax import entmax
    from src.policy.sparsemax import sparsemax

    z = torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0]])
    soft = torch.softmax(z, dim=-1)
    mid = entmax(z, alpha=1.5)
    hard = sparsemax(z)
    soft_nz = int((soft > 1e-8).sum())
    mid_nz = int((mid > 1e-8).sum())
    hard_nz = int((hard > 1e-8).sum())
    assert soft_nz >= mid_nz >= hard_nz
    assert torch.allclose(mid.sum(dim=-1), torch.ones(1), atol=1e-5)
    assert (mid >= -1e-8).all()


def test_entmax_batch_and_backward():
    from src.policy.entmax import entmax

    z = torch.randn(4, 8, requires_grad=True)
    w = entmax(z, alpha=1.5)
    assert w.shape == z.shape
    assert torch.allclose(w.sum(dim=-1), torch.ones(4), atol=1e-5)
    loss = w.sum()
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_tsallis_entropy_alpha2_uniform_and_onehot():
    from src.policy.entmax import tsallis_entropy

    k = 100
    uniform = torch.full((1, k), 1.0 / k)
    onehot = torch.zeros(1, k)
    onehot[0, 0] = 1.0
    h_u = float(tsallis_entropy(uniform, alpha=2.0).item())
    h_o = float(tsallis_entropy(onehot, alpha=2.0).item())
    assert h_u == pytest.approx(1.0 - 1.0 / k, **FLOAT_TOL)
    assert h_o == pytest.approx(0.0, **FLOAT_TOL)


def test_tsallis_entropy_zeros_safe():
    from src.policy.entmax import tsallis_entropy

    p = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    h = tsallis_entropy(p, alpha=2.0)
    assert torch.isfinite(h).all()
    assert float(h.item()) == pytest.approx(1.0 - 0.5, **FLOAT_TOL)


def test_apply_weight_head_entmax_15_allows_zeros():
    from src.policy.single_agent import _apply_weight_head

    raw = torch.tensor([[2.0, -2.0, 1.0, -3.0]])
    w_base = torch.full((4,), 0.25)
    w = _apply_weight_head(
        raw, "entmax_15", tilt_gain=5.0, temperature=1.0, w_base=w_base
    )
    assert w.shape == (1, 4)
    assert torch.allclose(w.sum(dim=-1), torch.ones(1), atol=1e-5)
    assert float((w <= 1e-8).sum()) >= 1 or float(w.max()) > 0.25 + 1e-4


def test_apply_weight_head_sparse_tilt_tsallis_matches_sparse_tilt():
    from src.policy.single_agent import _apply_weight_head

    raw = torch.tensor([[1.0, -1.0, 0.5, -0.5]])
    w_base = torch.full((4,), 0.25)
    a = _apply_weight_head(
        raw, "sparse_tilt", tilt_gain=5.0, temperature=1.0, w_base=w_base
    )
    b = _apply_weight_head(
        raw, "sparse_tilt_tsallis", tilt_gain=5.0, temperature=1.0, w_base=w_base
    )
    assert torch.allclose(a, b, atol=1e-6)


def test_ppo_sparse_tilt_tsallis_entropy_gate_differs():
    from src.policy.single_agent import PPOAgent

    torch.manual_seed(0)
    agent_gauss = PPOAgent(
        obs_dim=8,
        action_dim=4,
        weight_head="sparse_tilt",
        weight_head_tilt_gain=5.0,
        hidden=32,
        clip_eps=0.3,
        entropy_coef=0.01,
    )
    agent_tsallis = PPOAgent(
        obs_dim=8,
        action_dim=4,
        weight_head="sparse_tilt_tsallis",
        weight_head_tilt_gain=5.0,
        hidden=32,
        clip_eps=0.3,
        entropy_coef=0.01,
    )
    # Share weights so only entropy path differs.
    agent_tsallis.net.load_state_dict(agent_gauss.net.state_dict())
    agent_gauss._last_w_base = torch.full((4,), 0.25)
    agent_tsallis._last_w_base = torch.full((4,), 0.25)

    obs = torch.randn(16, 8)
    actions = torch.randn(16, 4)
    rewards = torch.randn(16)
    next_obs = torch.randn(16, 8)
    dones = torch.zeros(16)
    old_logprobs = torch.randn(16)

    stats_g = agent_gauss.train_epoch(
        obs=obs,
        actions=actions,
        rewards=rewards,
        next_obs=next_obs,
        dones=dones,
        old_logprobs=old_logprobs,
        n_epochs=1,
        n_minibatches=2,
    )
    stats_t = agent_tsallis.train_epoch(
        obs=obs,
        actions=actions,
        rewards=rewards,
        next_obs=next_obs,
        dones=dones,
        old_logprobs=old_logprobs,
        n_epochs=1,
        n_minibatches=2,
    )
    # Entropy metrics must differ when the Tsallis gate fires.
    assert "entropy" in stats_g and "entropy" in stats_t
    assert abs(float(stats_g["entropy"]) - float(stats_t["entropy"])) > 1e-6


def test_registry_allows_new_heads():
    from src.spectrum.registry import allowed_weight_heads, validate_cfg

    allowed = allowed_weight_heads("ppo")
    assert "entmax_15" in allowed
    assert "sparse_tilt_tsallis" in allowed

    cfg = {
        "portfolio_arm": "eq",
        "algo": "ppo",
        "policy_algo": "ppo",
        "architecture": "mlp",
        "temporal_backend": "mlp",
        "objective": "mean_std_cao",
        "train_world": "historical",
        "train_distribution": "historical",
        "weight_head": "entmax_15",
        "head_axis_id": "entmax_15",
        "action_law": "entmax_15",
        "claim_tier": "research",
        "protocol_tier": "screening",
        "n_assets": 100,
        "rebalance_cadence": "daily",
        "projection_mode": "soft",
        "use_equity_feature_cube": True,
        "primary_train": "historical_arm_env",
        "grid_kind": "cherrypick_rc6",
    }
    out = validate_cfg(cfg)
    assert out["algo"] == "ppo"


def test_cell_schema_allows_new_heads():
    from src.spectrum.cell_schema import ALLOWED_WEIGHT_HEADS, validate_cell_cfg

    assert "entmax_15" in ALLOWED_WEIGHT_HEADS
    assert "sparse_tilt_tsallis" in ALLOWED_WEIGHT_HEADS
    cfg = {
        "portfolio_arm": "eq",
        "algo": "ppo",
        "policy_algo": "ppo",
        "architecture": "mlp",
        "temporal_backend": "mlp",
        "objective": "mean_std_cao",
        "train_world": "historical",
        "train_distribution": "historical",
        "weight_head": "sparse_tilt_tsallis",
        "head_axis_id": "sparse_tilt_tsallis",
        "action_law": "sparse_tilt_tsallis",
        "claim_tier": "research",
        "protocol_tier": "screening",
        "n_assets": 100,
        "policy_mode": "shared",
        "agent": "single",
        "policy": "single_agent",
        "seeds": [0],
        "train_env_steps": 300000,
        "cpcv_n_splits": 6,
        "cpcv_n_test_groups": 2,
        "cpcv_purge_days": 21,
        "cpcv_embargo_days": 21,
        "selection_start": "2003-01-01",
        "selection_end": "2012-12-31",
        "oos_start": "2014-01-01",
        "oos_end": "2024-12-31",
        "spectrum_cell_id": "eq_K100_single_ppo_mlp_sparse_tilt_tsallis_mean_std_cao",
    }
    validate_cell_cfg(cfg)
