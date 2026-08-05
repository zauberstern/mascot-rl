"""CPPO trajectory CVaR constraint updates."""
from __future__ import annotations

import torch

from mascotrl.policy.cppo import CPPOAgent


def test_cppo_duals_update_after_train_epoch():
    agent = CPPOAgent(8, 4, hidden=16, normalize_obs=False)
    obs = torch.randn(16, 8)
    actions = torch.randn(16, 4)
    rewards = torch.linspace(-0.05, 0.05, 16)
    stats = agent.train_epoch(
        obs=obs,
        actions=actions,
        rewards=rewards,
        next_obs=obs,
        dones=torch.zeros(16),
        old_logprobs=torch.zeros(16),
    )
    assert "cvar_eta" in stats
    assert "cvar_nu" in stats
    assert "trajectory_cvar" in stats


def test_cppo_nu_non_negative():
    agent = CPPOAgent(8, 4, hidden=16, normalize_obs=False)
    for _ in range(3):
        agent.train_epoch(
            obs=torch.randn(12, 8),
            actions=torch.randn(12, 4),
            rewards=torch.randn(12) * 0.1,
            next_obs=torch.randn(12, 8),
            dones=torch.zeros(12),
            old_logprobs=torch.zeros(12),
        )
    assert float(agent.nu.item()) >= 0.0
