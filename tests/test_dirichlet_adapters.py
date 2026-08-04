"""WP-P6: Dirichlet adapters stamp action_law; SAC uses Dirichlet entropy."""
from __future__ import annotations

import torch

from src.policy.single_agent import DDPGAgent, SACAgent, TD3Agent


def test_td3_ddpg_dirichlet_mean_action_law() -> None:
    for cls in (TD3Agent, DDPGAgent):
        agent = cls(obs_dim=6, action_dim=4, weight_head="dirichlet_mean")
        assert agent.action_law == "dirichlet_mean"
        assert not hasattr(agent, "act_and_logp_raw")
        w = agent.act(torch.randn(2, 6), deterministic=True)
        assert w.shape == (2, 4)
        assert abs(float(w[0].detach().sum()) - 1.0) < 1e-4
        assert float(w.detach().min()) >= -1e-6


def test_sac_dirichlet_entropy_uses_dirichlet() -> None:
    agent = SACAgent(obs_dim=6, action_dim=4, weight_head="dirichlet_entropy")
    assert agent.action_law == "dirichlet_entropy"
    obs = torch.randn(8, 6)
    with torch.no_grad():
        actions = agent.act(obs, deterministic=False)
    stats = agent.train_epoch(
        obs=obs,
        actions=actions.detach(),
        rewards=torch.randn(8),
        next_obs=torch.randn(8, 6),
        dones=torch.zeros(8),
    )
    assert stats.get("action_law") == "dirichlet_entropy"
    assert "entropy" in stats
