"""SB3 backbone adapter smoke tests."""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("stable_baselines3")
pytest.importorskip("gymnasium")

from mascotrl.policy.sb3_adapter import make_sb3_agent, resolve_rl_backend
from mascotrl.policy.single_agent import make_single_agent


def test_rl_backend_switch_custom_vs_sb3():
    assert resolve_rl_backend({"rl_backend": "custom"}) == "custom"
    assert resolve_rl_backend({}) == "sb3"


def test_sb3_ppo_finite_actions():
    agent = make_sb3_agent("ppo", obs_dim=12, action_dim=3, weight_head="tanh_l1")
    obs = torch.randn(1, 12)
    w = agent.act(obs, deterministic=True)
    assert w.shape == (1, 3)
    assert torch.isfinite(w).all()


def test_make_single_agent_rl_backend_custom_fallback():
    agent = make_single_agent(
        "ppo", obs_dim=8, action_dim=2, rl_backend="custom", hidden=16, normalize_obs=False
    )
    assert getattr(agent, "name", "") == "ppo"
    obs = torch.randn(1, 8)
    assert agent.act(obs).shape[0] == 1
