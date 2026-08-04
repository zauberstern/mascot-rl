"""RL backend default must be SB3 when supported; custom when forced."""
from __future__ import annotations

import pytest

from src.policy.sb3_adapter import resolve_rl_backend
from src.policy.single_agent import make_single_agent


def test_resolve_rl_backend_default_sb3():
    assert resolve_rl_backend(None) == "sb3"
    assert resolve_rl_backend({}) == "sb3"
    assert resolve_rl_backend({"rl_backend": None}) == "sb3"
    assert resolve_rl_backend({"rl_backend": "custom"}) == "custom"
    assert resolve_rl_backend({"rl_backend": "SB3"}) == "sb3"


def test_make_single_agent_default_sb3_ppo():
    agent = make_single_agent("ppo", obs_dim=8, action_dim=4, lr=1e-3)
    assert getattr(agent, "backend", None) == "sb3"
    assert agent.name == "ppo"


def test_make_single_agent_force_custom_ppo():
    agent = make_single_agent("ppo", obs_dim=8, action_dim=4, lr=1e-3, rl_backend="custom")
    assert getattr(agent, "backend", "custom") != "sb3" or agent.__class__.__name__ == "PPOAgent"
    # Custom PPOAgent has no SB3 _model attribute contract
    assert agent.name == "ppo"
    assert not hasattr(agent, "_model") or agent.__class__.__module__.endswith("single_agent")
