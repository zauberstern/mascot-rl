"""Phase 1 systemic hardening: adapter interface regressions."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.plumbing
import torch

from src.policy.single_agent import make_single_agent


def test_sb3_recurrent_train_epoch_raises_not_implemented():
    """Direct SB3 RecurrentPPO must refuse stub training."""
    pytest.importorskip("sb3_contrib")
    from src.policy.sb3_adapter import make_sb3_agent

    agent = make_sb3_agent(
        "ppo_recurrent", obs_dim=12, action_dim=2, num_assets=2, seq_len=2
    )
    with pytest.raises(NotImplementedError, match="rl_backend=custom"):
        agent.train_epoch(
            obs=torch.randn(8, 12),
            actions=torch.randn(8, 2),
            rewards=torch.randn(8),
            next_obs=torch.randn(8, 12),
            dones=torch.zeros(8),
        )


def test_make_single_agent_gru_sb3_falls_back_to_custom():
    """PPO+gru under sb3 must fall back to custom (RecurrentPPO train is stub)."""
    agent = make_single_agent(
        "ppo",
        obs_dim=12,
        action_dim=2,
        lr=1e-3,
        rl_backend="sb3",
        architecture="gru",
        num_assets=2,
        seq_len=2,
        d_model=3,
    )
    assert getattr(agent, "backend", "custom") == "custom"
    assert agent.name == "ppo"


def test_sb3_dqn_no_act_and_logp_raw():
    agent = make_single_agent("dqn", obs_dim=8, action_dim=2, lr=1e-3, rl_backend="sb3", n_bins=3)
    assert not hasattr(agent, "act_and_logp_raw")


def test_custom_dqn_act_and_logp_raw_returns_levels():
    agent = make_single_agent("dqn", obs_dim=8, action_dim=4, lr=1e-3, rl_backend="custom")
    obs = torch.randn(3, 8)
    raw, logp = agent.act_and_logp_raw(obs, deterministic=True)
    assert raw.shape == (3, 4)
    assert logp.shape == (3,)
    # Raw levels must be in the discrete set before L1.
    for v in raw.reshape(-1).tolist():
        assert v in (-1.0, 0.0, 1.0)
    w = agent.act(obs, deterministic=True)
    assert torch.isfinite(w).all()


def test_sb3_opt_property_exposes_policy_optimizer():
    agent = make_single_agent("ppo", obs_dim=6, action_dim=3, lr=1e-3, rl_backend="sb3")
    opt = agent.opt
    assert opt is not None
    assert hasattr(opt, "state_dict")


def test_sb3_ppo_optimizer_steps_are_per_call():
    """train_epoch must report per-call steps, not cumulative."""
    agent = make_single_agent("ppo", obs_dim=8, action_dim=4, lr=1e-3, rl_backend="sb3")
    t = 16
    batch = dict(
        obs=torch.randn(t, 8),
        actions=torch.randn(t, 4),
        rewards=torch.randn(t),
        next_obs=torch.randn(t, 8),
        dones=torch.zeros(t),
        old_logprobs=torch.zeros(t),
        n_epochs=1,
        n_minibatches=1,
    )
    s1 = agent.train_epoch(**batch)
    s2 = agent.train_epoch(**batch)
    # Per-call: each call reports steps taken in that call, not running total.
    assert s1["optimizer_steps"] >= 1.0
    assert s2["optimizer_steps"] == s1["optimizer_steps"]


def test_is_ppo_style_includes_cppo_omnisafe_and_ppo_recurrent():
    from src.eval.research_alpha_train import _is_ppo_style

    class _A:
        def __init__(self, name):
            self.name = name

    assert _is_ppo_style(_A("ppo"))
    assert _is_ppo_style(_A("cppo"))
    assert _is_ppo_style(_A("cppo_omnisafe"))
    assert _is_ppo_style(_A("ppo_recurrent"))
    assert not _is_ppo_style(_A("sac"))
    assert not _is_ppo_style(_A("mcpg"))


def test_mcpg_train_epoch_called_once_despite_n_minibatches(monkeypatch):
    """MCPG must not re-score the same on-policy batch n_minibatches times."""
    import numpy as np

    from src.eval.research_alpha_train import train_research_hist
    from src.policy.single_agent import MCPGAgent

    calls = {"n": 0}
    real = MCPGAgent.train_epoch

    def _spy(self, **kwargs):
        calls["n"] += 1
        return real(self, **kwargs)

    monkeypatch.setattr(MCPGAgent, "train_epoch", _spy)
    rng = np.random.default_rng(0)
    t, k = 24, 2
    rets = rng.normal(0.0002, 0.01, size=(t, k))
    fac = rng.normal(0.0, 0.01, size=(t, 4))
    cfg = {
        "primary_train": "historical_arm_env",
        "portfolio_arm": "eq",
        "n_assets": k,
        "train_epochs": 1,
        "n_minibatches": 4,
        "policy": "single_agent",
        "projection_mode": "soft",
        "algo": "mcpg",
        "objective": "mtm_pnl",
        "rl_backend": "custom",
        "lr": 1e-3,
    }
    out = train_research_hist(rets, fac, cfg, seed=0)
    assert calls["n"] == 1, f"MCPG train_epoch called {calls['n']} times; want 1"
    assert out["train_stats"].get("on_policy_single_shot") is True
