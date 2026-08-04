"""Checkpoint save/load must reproduce identical deterministic forward pass."""
from __future__ import annotations

import pytest
import torch

from src.policy.single_agent import make_single_agent


@pytest.mark.integration
@pytest.mark.parametrize("algo", ["ppo", "sac", "td3", "ddpg"])
def test_checkpoint_save_load_identical_forward(algo, torch_deterministic, tmp_path):
    """Save agent, reload, same obs = identical action (bit-exact)."""
    obs_dim, action_dim = 8, 4
    agent = make_single_agent(
        algo, obs_dim=obs_dim, action_dim=action_dim, lr=1e-3, rl_backend="custom"
    )
    obs = torch.randn(1, obs_dim)
    w_before = agent.act(obs, deterministic=True)
    state = agent.checkpoint_state()
    path = tmp_path / "ckpt.pt"
    torch.save(state, path)

    agent2 = make_single_agent(
        algo, obs_dim=obs_dim, action_dim=action_dim, lr=1e-3, rl_backend="custom"
    )
    agent2.load_checkpoint_state(torch.load(path, weights_only=False))
    w_after = agent2.act(obs, deterministic=True)
    torch.testing.assert_close(w_before, w_after, atol=0.0, rtol=0.0)
