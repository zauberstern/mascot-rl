"""Smoke-train single-agent PPO/SAC/TD3 adapters (Alpha v2 Block D Step 18)."""
from __future__ import annotations

import torch
import pytest

from src.policy.single_agent import (
    DDPGAgent,
    DQNAgent,
    MCPGAgent,
    PPOAgent,
    RRLAgent,
    SACAgent,
    TD3Agent,
    make_single_agent,
)


@pytest.mark.parametrize(
    "algo,cls",
    [
        ("ppo", PPOAgent),
        ("sac", SACAgent),
        ("td3", TD3Agent),
        ("ddpg", DDPGAgent),
        ("mcpg", MCPGAgent),
        ("rrl", RRLAgent),
        ("dqn", DQNAgent),
    ],
)
def test_single_agent_smoke_one_epoch(algo: str, cls):
    torch.manual_seed(0)
    obs_dim, action_dim = 16, 8
    # Custom path: these tests assert hand-rolled agent classes.
    agent = make_single_agent(
        algo, obs_dim=obs_dim, action_dim=action_dim, lr=1e-3, rl_backend="custom"
    )
    assert isinstance(agent, cls)

    batch_n = 32
    obs = torch.randn(batch_n, obs_dim)
    actions = torch.randn(batch_n, action_dim)
    rewards = torch.randn(batch_n)
    next_obs = torch.randn(batch_n, obs_dim)
    dones = torch.zeros(batch_n)

    # Shared obs/action interface.
    a0 = agent.act(obs[:4], deterministic=True)
    assert a0.shape == (4, action_dim)
    assert torch.all(torch.isfinite(a0))

    stats = agent.train_epoch(
        obs=obs,
        actions=actions,
        rewards=rewards,
        next_obs=next_obs,
        dones=dones,
    )
    assert isinstance(stats, dict)
    assert "loss" in stats or "actor_loss" in stats or "policy_loss" in stats
    assert all(
        (not isinstance(v, float)) or (v == v)  # finite if float
        for v in stats.values()
        if isinstance(v, (int, float))
    )
