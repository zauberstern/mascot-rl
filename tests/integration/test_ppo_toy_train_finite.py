"""Short PPO toy train: losses and weights stay finite."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mascotrl.policy.single_agent import make_single_agent


@pytest.mark.integration
def test_ppo_toy_train_finite(torch_deterministic):
    """Few PPO updates on a tiny batch must keep losses/weights finite."""
    obs_dim, action_dim = 8, 3
    agent = make_single_agent(
        "ppo", obs_dim=obs_dim, action_dim=action_dim, lr=1e-3, rl_backend="custom"
    )
    batch_n = 64
    obs = torch.randn(batch_n, obs_dim)
    actions = torch.randn(batch_n, action_dim)
    rewards = torch.randn(batch_n)
    next_obs = torch.randn(batch_n, obs_dim)
    dones = torch.zeros(batch_n)

    for _ in range(3):
        stats = agent.train_epoch(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
        )
        for k, v in stats.items():
            if isinstance(v, float):
                assert np.isfinite(v), f"{k}={v}"

    a = agent.act(obs[:4], deterministic=True)
    assert torch.all(torch.isfinite(a))
    assert a.shape == (4, action_dim)
