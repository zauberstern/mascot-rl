"""HARL HAPPO adapter conformance tests."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("harl")
pytest.importorskip("gymnasium")

from mascotrl.policy.harl_adapter import (
    HARLHAPPOBundle,
    HistoricalArmHARLEnv,
    default_happo_args,
    resolve_use_harl,
)


class _ToyHistEnv:
    """Minimal hist-env stub: obs = ones, reward = -||w||^2 coordination."""

    def __init__(self, k: int = 2):
        self.K = k
        self.obs_dim = k * 4
        self._t = 0

    def reset(self, seed=None):
        self._t = 0
        return np.ones(self.obs_dim, dtype=np.float32), {}

    def step(self, w):
        self._t += 1
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        # Cooperative: prefer equal weights
        target = np.full(self.K, 1.0 / self.K)
        reward = -float(np.sum((w - target) ** 2))
        done = self._t >= 8
        return np.ones(self.obs_dim, dtype=np.float32), reward, done, False, {}


def test_resolve_use_harl():
    assert resolve_use_harl({}) is False
    assert resolve_use_harl({"use_harl": True}) is True


def test_harl_bundle_constructs_and_acts():
    bundle = HARLHAPPOBundle(2, obs_dim_per_agent=4, args=default_happo_args(ppo_epoch=1))
    assert bundle.n_agents == 2
    order = bundle.sequential_agent_order()
    assert order == [0, 1]
    obs = [np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32)]
    acts = bundle.act_all(obs, deterministic=True)
    assert len(acts) == 2
    assert all(a.size >= 1 for a in acts)


def test_harl_env_interface_and_sequential_order():
    env = HistoricalArmHARLEnv(_ToyHistEnv(2), n_agents=2)
    assert env.n_agents == 2
    obs, share, avail = env.reset()
    assert len(obs) == 2
    assert share.ndim == 1
    bundle = HARLHAPPOBundle(2, obs_dim_per_agent=env._per_agent_dim)
    rewards = []
    for _ in range(5):
        order = bundle.sequential_agent_order()
        assert order[0] == 0 and order[1] == 1  # default sequential
        acts = bundle.act_all(obs, deterministic=False)
        # re-order actions by agent id (already in agent order)
        obs, share, rews, dones, infos, avail = env.step(acts)
        rewards.append(float(rews[0][0]))
        if dones[0]:
            break
    assert len(rewards) >= 1
    assert all(np.isfinite(r) for r in rewards)


def test_harl_vs_custom_flag_off_by_default():
    """Custom HAPPO remains default; HARL is opt-in."""
    assert resolve_use_harl({"algo": "happo"}) is False
