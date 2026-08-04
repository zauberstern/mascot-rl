"""Agent train_epoch stats contract: required keys, string rl_backend, no NaN."""
from __future__ import annotations

import math

import pytest
import torch

from src.policy.single_agent import make_single_agent


_REQUIRED = ("loss", "rl_backend", "optimizer_steps")


def _finite_stats(stats: dict) -> None:
    for k, v in stats.items():
        if isinstance(v, float):
            assert not math.isnan(v), f"{k} is NaN"
        if k == "rl_backend":
            assert isinstance(v, str), f"rl_backend must be str, got {type(v)}"


@pytest.mark.parametrize("algo", ["ppo", "sac", "td3", "ddpg"])
def test_sb3_train_epoch_stats_contract(algo: str):
    agent = make_single_agent(algo, obs_dim=8, action_dim=4, lr=1e-3, rl_backend="sb3")
    t = 16
    obs = torch.randn(t, 8)
    actions = torch.randn(t, 4)
    rewards = torch.randn(t)
    next_obs = torch.randn(t, 8)
    dones = torch.zeros(t)
    stats = agent.train_epoch(
        obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones
    )
    for key in _REQUIRED:
        assert key in stats, f"missing {key} for {algo}"
    assert stats["rl_backend"] == "sb3"
    _finite_stats(stats)


def test_custom_ppo_train_epoch_has_loss():
    agent = make_single_agent("ppo", obs_dim=8, action_dim=4, lr=1e-3, rl_backend="custom")
    t = 16
    obs = torch.randn(t, 8)
    actions = torch.randn(t, 4)
    rewards = torch.randn(t)
    next_obs = torch.randn(t, 8)
    dones = torch.zeros(t)
    old_logp = torch.zeros(t)
    stats = agent.train_epoch(
        obs=obs,
        actions=actions,
        rewards=rewards,
        next_obs=next_obs,
        dones=dones,
        old_logprobs=old_logp,
    )
    assert "loss" in stats or "actor_loss" in stats or "policy_loss" in stats
    _finite_stats({k: v for k, v in stats.items() if isinstance(v, (int, float, str))})


def test_sb3_offpolicy_act_and_logp_raw_no_get_distribution():
    """SAC/TD3/DDPG must not call policy.get_distribution (PPO-only API)."""
    for algo in ("sac", "td3", "ddpg"):
        agent = make_single_agent(algo, obs_dim=8, action_dim=4, lr=1e-3, rl_backend="sb3")
        obs = torch.randn(2, 8)
        raw, logp = agent.act_and_logp_raw(obs, deterministic=False)
        assert raw.shape == (2, 4)
        assert logp.shape == (2,)
        assert torch.isfinite(raw).all()
        assert torch.isfinite(logp).all()
        w = agent.raw_to_weights(raw)
        assert w.shape == (2, 4)


@pytest.mark.parametrize("algo", ["sac", "td3", "ddpg", "dqn"])
def test_sb3_offpolicy_train_epoch_past_learning_starts(algo: str):
    """Replay fill past learning_starts must train without AttributeError.

    SB3 OffPolicyAlgorithm.train() requires ``_logger`` from ``_setup_learn``.
    """
    kwargs: dict = {"obs_dim": 8, "action_dim": 4, "lr": 1e-3, "rl_backend": "sb3"}
    if algo == "dqn":
        kwargs["n_bins"] = 3
    agent = make_single_agent(algo, **kwargs)
    t = 64  # > default learning_starts=32
    obs = torch.randn(t, 8)
    actions = torch.randn(t, 4)
    rewards = torch.randn(t)
    next_obs = torch.randn(t, 8)
    dones = torch.zeros(t)
    stats = agent.train_epoch(
        obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones
    )
    assert "loss" in stats
    assert stats["optimizer_steps"] >= 1.0
    assert stats.get("loss_source") != "stub"
    _finite_stats(stats)


def test_sb3_recurrent_ppo_no_act_and_logp_raw():
    """RecurrentPPO must not expose act_and_logp_raw (needs lstm_states).

    research_alpha_train falls back to act() when hasattr is False.
    Construct via make_sb3_agent directly (make_single_agent falls back to custom).
    """
    pytest.importorskip("sb3_contrib")
    from src.policy.sb3_adapter import make_sb3_agent

    agent = make_sb3_agent(
        "ppo_recurrent", obs_dim=12, action_dim=2, num_assets=2, seq_len=2
    )
    assert not hasattr(agent, "act_and_logp_raw")
    obs = torch.randn(1, 12)
    w = agent.act(obs, deterministic=False)
    assert w.shape[0] == 1
    assert torch.isfinite(w).all()


def test_sb3_recurrent_train_research_hist_no_typeerror():
    """GRU under default sb3 falls back to custom (RecurrentPPO train stubbed)."""
    import numpy as np

    from src.eval.research_alpha_train import train_research_hist

    rng = np.random.default_rng(0)
    t, k = 24, 2
    rets = rng.normal(0.0002, 0.01, size=(t, k))
    fac = rng.normal(0.0, 0.01, size=(t, 4))
    cfg = {
        "primary_train": "historical_arm_env",
        "portfolio_arm": "eq",
        "n_assets": k,
        "train_epochs": 1,
        "policy": "single_agent",
        "projection_mode": "soft",
        "use_equity_feature_cube": True,
        "architecture": "gru",
        "algo": "ppo",
        "rl_backend": "sb3",
        "lr": 1e-3,
    }
    out = train_research_hist(rets, fac, cfg, seed=0)
    assert out["n_steps"] > 0
    assert getattr(out["agent"], "backend", "custom") == "custom"
    assert hasattr(out["agent"], "act_and_logp_raw")
