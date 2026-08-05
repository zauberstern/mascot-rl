"""SB3 vs custom interface parity: collect/train/checkpoint contract.

Guards against shipping another get_distribution / logger / net mismatch.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

pytest.importorskip("stable_baselines3")
pytest.importorskip("gymnasium")

from mascotrl.eval.research_alpha_train import train_research_hist
from mascotrl.policy.single_agent import make_single_agent

_COLLECT_TRAIN = ("act", "train_epoch")
_ARTIFACT_KEYS = (
    "mean_reward",
    "n_steps",
    "rl_backend",
    "optimizer_steps",
    "train_stats",
)


def _toy_panel(t: int = 24, k: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, 0.01, size=(t, k))
    factors = rng.normal(0.0, 0.01, size=(t, 4))
    return rets, factors


@pytest.mark.parametrize(
    "algo,extra",
    [
        ("ppo", {}),
        ("sac", {}),
        ("td3", {}),
        ("ddpg", {}),
        ("dqn", {"n_bins": 3}),
        ("ppo_recurrent", {"num_assets": 2, "seq_len": 2}),
    ],
)
def test_sb3_hasattr_parity_collect_train_net(algo: str, extra: dict):
    """Research collect/train/checkpoint surface must exist on every SB3 algo."""
    if algo == "ppo_recurrent":
        pytest.importorskip("sb3_contrib")
        from mascotrl.policy.sb3_adapter import make_sb3_agent

        agent = make_sb3_agent("ppo_recurrent", obs_dim=12, action_dim=2, **extra)
    else:
        agent = make_single_agent(
            algo, obs_dim=8, action_dim=2, lr=1e-3, rl_backend="sb3", **extra
        )
    for name in _COLLECT_TRAIN:
        assert hasattr(agent, name), f"sb3 {algo} missing {name}"
    assert hasattr(agent, "net"), f"sb3 {algo} missing net (checkpoint)"
    assert isinstance(agent.net, torch.nn.Module)
    if algo != "ppo_recurrent" and algo != "dqn":
        assert hasattr(agent, "raw_to_weights")
        assert hasattr(agent, "act_and_logp_raw")
    if algo in ("ppo_recurrent", "dqn"):
        assert not hasattr(agent, "act_and_logp_raw")
    # freeze_obs_norm is optional (custom PPO only)
    _ = getattr(agent, "freeze_obs_norm", None)


@pytest.mark.parametrize("algo", ["ppo", "sac", "td3", "ddpg"])
def test_sb3_custom_train_research_hist_smoke_parity(algo: str):
    """Both backends return finite mean_reward and the same artifact keys."""
    rets, fac = _toy_panel(t=24, k=2)
    # Off-policy algos refuse episode-return objectives (mean_std_cao default).
    objective = "mtm_pnl" if algo != "ppo" else "mean_std_cao"
    base = {
        "primary_train": "historical_arm_env",
        "portfolio_arm": "eq",
        "n_assets": 2,
        "train_epochs": 1,
        "policy": "single_agent",
        "projection_mode": "soft",
        "algo": algo,
        "objective": objective,
        "lr": 1e-3,
        "weight_head": "softmax" if algo == "ppo" else "tanh_l1",
    }
    outs = {}
    for backend in ("sb3", "custom"):
        outs[backend] = train_research_hist(
            rets, fac, dict(base, rl_backend=backend), seed=0
        )
    for backend, out in outs.items():
        for key in _ARTIFACT_KEYS:
            assert key in out, f"{backend} missing artifact key {key}"
        mr = float(out["mean_reward"])
        assert math.isfinite(mr), f"{backend} mean_reward not finite: {mr}"
        assert out["rl_backend"] == backend
        assert out["n_steps"] > 0


def test_sb3_offpolicy_train_epoch_large_batch_no_raise():
    """train_epoch with batch size > 32 must not raise for off-policy SB3."""
    for algo in ("sac", "td3", "ddpg"):
        agent = make_single_agent(algo, obs_dim=8, action_dim=4, lr=1e-3, rl_backend="sb3")
        t = 96
        stats = agent.train_epoch(
            obs=torch.randn(t, 8),
            actions=torch.randn(t, 4),
            rewards=torch.randn(t),
            next_obs=torch.randn(t, 8),
            dones=torch.zeros(t),
        )
        assert stats["optimizer_steps"] >= 1.0
        assert stats.get("loss_source") != "stub"


def test_sb3_recurrent_train_research_hist_smoke():
    """GRU under rl_backend=sb3 falls back to custom (RecurrentPPO train stubbed)."""
    rets, fac = _toy_panel(t=24, k=2)
    cfg = {
        "primary_train": "historical_arm_env",
        "portfolio_arm": "eq",
        "n_assets": 2,
        "train_epochs": 1,
        "policy": "single_agent",
        "projection_mode": "soft",
        "algo": "ppo",
        "architecture": "gru",
        "use_equity_feature_cube": True,
        "rl_backend": "sb3",
        "lr": 1e-3,
    }
    out = train_research_hist(rets, fac, cfg, seed=0)
    assert math.isfinite(float(out["mean_reward"]))
    assert out["n_steps"] > 0
    assert getattr(out["agent"], "backend", "custom") == "custom"
    for key in _ARTIFACT_KEYS:
        assert key in out
