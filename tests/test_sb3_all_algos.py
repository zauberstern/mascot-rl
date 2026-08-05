"""Expanded SB3 coverage: all algos, DQN MultiDiscrete, RecurrentPPO."""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("stable_baselines3")
pytest.importorskip("gymnasium")

from mascotrl.policy.sb3_adapter import (
    PortfolioFeaturesExtractor,
    make_sb3_agent,
    resolve_rl_backend,
)


@pytest.mark.parametrize("algo", ["ppo", "sac", "td3", "ddpg"])
def test_sb3_all_algos_finite_actions(algo):
    agent = make_sb3_agent(algo, obs_dim=12, action_dim=3)
    obs = torch.randn(1, 12)
    w = agent.act(obs, deterministic=True)
    assert w.shape[0] == 1
    assert torch.isfinite(w).all()


def test_sb3_dqn_multidiscrete():
    agent = make_sb3_agent("dqn", obs_dim=8, action_dim=4, n_bins=3)
    obs = torch.randn(1, 8)
    w = agent.act(obs, deterministic=True)
    assert w.shape == (1, 4)
    assert torch.isfinite(w).all()
    # Centered levels {-1,0,1} then L1-normalize (matches custom DQN; may be 0).
    l1 = float(w.abs().sum())
    assert l1 == pytest.approx(1.0, abs=1e-5) or l1 == pytest.approx(0.0, abs=1e-5)


def test_sb3_dqn_act_can_produce_negative_weights():
    """Bin indices must center to {-1,...,1} so shorts are expressible."""
    agent = make_sb3_agent("dqn", obs_dim=8, action_dim=4, n_bins=3)
    # Force decode of all-zero flat action -> bins [0,0,0,0] -> levels [-1,...]
    # and all-max flat -> [2,2,2,2] -> [+1,...]; mix via predict over many obs.
    found_neg = False
    found_pos = False
    for seed in range(64):
        torch.manual_seed(seed)
        obs = torch.randn(4, 8)
        w = agent.act(obs, deterministic=False)
        assert w.shape == (4, 4)
        assert torch.isfinite(w).all()
        if (w < -1e-6).any():
            found_neg = True
        if (w > 1e-6).any():
            found_pos = True
        # L1 unit when any nonzero exposure
        for row in w:
            s = float(row.abs().sum())
            if s > 1e-6:
                assert s == pytest.approx(1.0, abs=1e-5)
    # Also verify decode centering explicitly
    levels0 = agent._bins_to_levels(agent._decode(0))
    assert float(levels0.min()) < 0.0
    levels_max = agent._bins_to_levels(agent._decode(agent._n_flat - 1))
    assert float(levels_max.max()) > 0.0
    assert found_neg or float(levels0.min()) < 0.0
    assert found_pos or float(levels_max.max()) > 0.0


def test_sb3_recurrent_extractor_reshape():
    pytest.importorskip("sb3_contrib")
    ext = PortfolioFeaturesExtractor(features_dim=32, num_assets=2, seq_len=2, n_channels=3)
    import gymnasium as gym
    import numpy as np

    space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
    module = ext.cls(space)
    x = torch.randn(4, 12)
    y = module(x)
    assert y.shape == (4, 32)


def test_sb3_recurrent_agent_smoke():
    pytest.importorskip("sb3_contrib")
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
    w = agent.act(torch.randn(1, 12))
    assert torch.isfinite(w).all()


def test_resolve_rl_backend_default_sb3():
    assert resolve_rl_backend({}) == "sb3"
    assert resolve_rl_backend({"rl_backend": "custom"}) == "custom"
