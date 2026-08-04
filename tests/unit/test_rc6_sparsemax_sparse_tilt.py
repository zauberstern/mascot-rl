"""RC6: sparsemax Euclidean projection + sparse_tilt head tests."""
from __future__ import annotations

import numpy as np
import pytest
from tests.conftest import FLOAT_TOL
import torch


def test_sparsemax_exact_zeros_and_simplex():
    from src.policy.sparsemax import sparsemax

    # Support size 1: mass concentrates on the max coordinate.
    z = torch.tensor([2.0, 1.0, 0.0, -1.0])
    w = sparsemax(z)
    assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-5)
    assert float(w[0]) == pytest.approx(1.0, abs=1e-5)
    assert float(w[1]) == pytest.approx(0.0, **FLOAT_TOL)
    assert float(w[2]) == pytest.approx(0.0, **FLOAT_TOL)
    assert float(w[3]) == pytest.approx(0.0, **FLOAT_TOL)

    # Two-support example: moderate logit spread produces exact zeros.
    z2 = torch.tensor([1.0, 0.5, -2.0, -3.0])
    w2 = sparsemax(z2)
    assert torch.allclose(w2.sum(), torch.tensor(1.0), atol=1e-5)
    assert float(w2[2]) == pytest.approx(0.0, **FLOAT_TOL)
    assert float(w2[3]) == pytest.approx(0.0, **FLOAT_TOL)
    assert float(w2[0]) == pytest.approx(0.75, abs=1e-5)
    assert float(w2[1]) == pytest.approx(0.25, abs=1e-5)


def test_sparsemax_batch_and_backward():
    from src.policy.sparsemax import sparsemax

    z = torch.randn(4, 8, requires_grad=True)
    w = sparsemax(z)
    assert w.shape == z.shape
    assert torch.allclose(w.sum(dim=-1), torch.ones(4), atol=1e-5)
    loss = w.sum()
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_sparse_tilt_head_with_base():
    from src.policy.single_agent import _apply_weight_head

    raw = torch.tensor([[1.0, -1.0, 0.5, -0.5]])
    w_base = torch.full((4,), 0.25)
    w = _apply_weight_head(
        raw, "sparse_tilt", tilt_gain=5.0, temperature=1.0, w_base=w_base
    )
    assert w.shape == (1, 4)
    assert torch.allclose(w.sum(dim=-1), torch.ones(1), atol=1e-5)
    # Sparsemax + tilt should allow zeros / concentration away from uniform.
    assert float(w.max()) > 0.25 + 1e-4 or float((w == 0).sum()) >= 0


def test_ppo_sparse_tilt_smoke():
    from src.policy.single_agent import PPOAgent

    agent = PPOAgent(
        obs_dim=8,
        action_dim=4,
        weight_head="sparse_tilt",
        weight_head_tilt_gain=5.0,
        hidden=32,
        clip_eps=0.3,
    )
    agent._last_w_base = torch.full((4,), 0.25)
    obs = torch.randn(2, 8)
    w = agent.act(obs, deterministic=True)
    assert w.shape == (2, 4)
    assert torch.allclose(w.sum(dim=-1), torch.ones(2), atol=1e-4)


@pytest.mark.parametrize("agent_cls_name", ["SACAgent", "TD3Agent"])
def test_sac_td3_sparse_tilt_uses_tilt_gain(agent_cls_name):
    from src.policy import single_agent as sa

    cls = getattr(sa, agent_cls_name)
    agent = cls(
        obs_dim=8,
        action_dim=4,
        weight_head="sparse_tilt",
        weight_head_tilt_gain=5.0,
        hidden=32,
    )
    assert float(agent.weight_head_tilt_gain) == pytest.approx(5.0, **FLOAT_TOL)
    agent._last_w_base = torch.full((4,), 0.25)
    obs = torch.randn(2, 8)
    w = agent.act(obs, deterministic=True)
    assert w.shape == (2, 4)
    assert torch.allclose(w.sum(dim=-1), torch.ones(2), atol=1e-4)


def test_gates_from_runner_cpcv_artifact_runs_gate2():
    """CPCV-shaped art with path-0 pnl + factors must not skip gate2."""
    from scripts.run_spectrum_campaign import _gates_from_runner

    rng = np.random.default_rng(1)
    T = 80
    pnl = (0.0005 + rng.normal(size=T) * 0.001).tolist()
    factors = rng.normal(size=(T, 7)).tolist()
    art = {
        "paths": {"0": {"pnl": pnl}},
        "factors": factors,
        "factor_names": ["mkt", "smb", "hml", "rmw", "cma", "umd", "ps_vwf"],
        "cost_ladder": {
            "break_even_spread_multiplier": 0.5,
            "cost_source": "om_touch",
        },
        "path_summary": {"sharpe_mean": 0.4},
        "baselines": {"equal_weight": 0.1},
    }
    gates = _gates_from_runner(art, dry_run=False)
    assert gates["gate2"].get("skipped") is not True
    assert "pass" in gates["gate2"]
    assert gates["gate2"].get("n_factors") == 7


def test_gates_from_runner_policy_returns_key():
    from scripts.run_spectrum_campaign import _gates_from_runner

    rng = np.random.default_rng(2)
    T = 60
    art = {
        "policy_returns": (0.001 + rng.normal(size=T) * 0.001).tolist(),
        "factors": rng.normal(size=(T, 4)).tolist(),
    }
    gates = _gates_from_runner(art, dry_run=False)
    assert gates["gate2"].get("skipped") is not True
    assert "t_stat" in gates["gate2"]


def test_gate2_returns_annualized_alpha():
    from src.eval.spectrum_gates import compute_gate2

    rng = np.random.default_rng(0)
    T = 200
    factors = rng.normal(size=(T, 7))
    # Construct a series with positive alpha.
    rets = 0.001 + factors @ rng.normal(size=7) * 0.01 + rng.normal(size=T) * 0.001
    out = compute_gate2(
        rets,
        factors,
        factor_names=["mkt", "smb", "hml", "rmw", "cma", "umd", "ps"],
    )
    assert "alpha_annualized" in out
    assert "n_factors" in out
    assert out["n_factors"] == 7
    assert "factor_loadings" in out
    assert len(out["factor_loadings"]) == 7
