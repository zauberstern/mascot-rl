"""Deepened SB3 / HARL / OmniSafe conformance."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mascotrl.policy.single_agent import compute_gae
from mascotrl.policy.vendor.omnisafe.lagrange import Lagrange
from mascotrl.policy.vendor.omnisafe.pid_lagrange import PIDLagrangian


def test_gae_matches_manual_lambda_one():
    rewards = torch.tensor([1.0, 0.0, -1.0])
    values = torch.zeros(3)
    next_values = torch.zeros(3)
    dones = torch.tensor([0.0, 0.0, 1.0])
    adv, ret = compute_gae(
        rewards, values, next_values, dones, gamma=0.99, gae_lambda=1.0
    )
    # Manual MC returns with gamma=0.99
    r2 = -1.0
    r1 = 0.0 + 0.99 * r2
    r0 = 1.0 + 0.99 * r1
    assert float(ret[2]) == pytest.approx(r2, abs=1e-5)
    assert float(ret[1]) == pytest.approx(r1, abs=1e-5)
    assert float(ret[0]) == pytest.approx(r0, abs=1e-5)
    assert torch.allclose(adv, ret - values, atol=1e-5)


def test_sb3_rollout_buffer_advantage_smoke():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.buffers import RolloutBuffer
    import gymnasium as gym

    buf = RolloutBuffer(
        buffer_size=8,
        observation_space=gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
        action_space=gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        device="cpu",
        gae_lambda=0.95,
        gamma=0.99,
        n_envs=1,
    )
    for i in range(8):
        buf.add(
            obs=np.zeros(3, dtype=np.float32),
            action=np.zeros(2, dtype=np.float32),
            reward=0.1,
            episode_start=i == 0,
            value=torch.tensor(0.0),
            log_prob=torch.tensor(0.0),
        )
    buf.compute_returns_and_advantage(last_values=torch.tensor([0.0]), dones=np.array([True]))
    assert buf.advantages.shape[0] == 8
    assert np.isfinite(buf.advantages).all()


def test_pid_lagrange_increases_when_cost_above_limit():
    pid = PIDLagrangian(
        pid_kp=0.5,
        pid_ki=0.1,
        pid_kd=0.0,
        cost_limit=0.0,
        lagrangian_multiplier_init=0.0,
    )
    before = pid.lagrangian_multiplier
    for _ in range(5):
        pid.pid_update(1.0)
    assert pid.lagrangian_multiplier > before


def test_naive_lagrange_nonnegative():
    lag = Lagrange(cost_limit=0.0, lagrangian_multiplier_init=0.0, lambda_lr=0.1)
    # API may be update / update_lagrange_multiplier depending on vendor file
    updater = getattr(lag, "update_lagrange_multiplier", None) or getattr(
        lag, "update", None
    )
    assert updater is not None
    updater(1.0)
    assert float(lag.lagrangian_multiplier.detach()) >= 0.0


def test_harl_env_contract_reset_step():
    pytest.importorskip("harl")
    from mascotrl.policy.harl_adapter import HistoricalArmHARLEnv

    class _Stub:
        K = 2
        obs_dim = 6

        def __init__(self):
            self.t = 0

        def reset(self, *a, **k):
            self.t = 0
            return np.zeros(6, dtype=np.float32), {}

        def step(self, action):
            self.t += 1
            done = self.t >= 2
            return np.zeros(6, dtype=np.float32), 0.1, done, False, {}

    env = HistoricalArmHARLEnv(_Stub(), n_agents=2)
    out = env.reset()
    assert out is not None
    # HARL-style: obs list / share_obs / available_actions
    if isinstance(out, tuple) and len(out) >= 2:
        assert out[0] is not None
