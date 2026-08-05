"""Phase C: correct PPO (real ratio, GAE, entropy, weight head)."""
from __future__ import annotations

import pytest
from tests.conftest import FLOAT_TOL

import numpy as np
import torch
import torch.nn.functional as F

from mascotrl.policy.single_agent import PPOAgent, make_single_agent


def test_ppo_ratio_not_identically_one_after_update():
    torch.manual_seed(0)
    agent = PPOAgent(obs_dim=4, action_dim=3, lr=1e-2, hidden=32, weight_head="softmax")
    T = 64
    obs = torch.randn(T, 4)
    with torch.no_grad():
        actions, logp_old = agent.act_and_logp(obs, deterministic=False)
    rewards = torch.randn(T)
    next_obs = torch.randn(T, 4)
    dones = torch.zeros(T)
    # First epoch: ratio starts near 1; after several updates ratio moves.
    stats = agent.train_epoch(
        obs=obs,
        actions=actions,
        rewards=rewards,
        next_obs=next_obs,
        dones=dones,
        old_logprobs=logp_old,
        n_epochs=4,
        n_minibatches=2,
    )
    assert "mean_ratio" in stats
    # After updates, mean_ratio should deviate from exactly 1.0.
    assert abs(float(stats["mean_ratio"]) - 1.0) > 1e-6 or float(stats["approx_kl"]) > 1e-8
    assert float(stats["optimizer_steps"]) >= 4


def test_ppo_action_sums_to_one_softmax():
    agent = PPOAgent(obs_dim=2, action_dim=5, weight_head="softmax")
    obs = torch.randn(8, 2)
    a = agent.act(obs, deterministic=True)
    assert a.shape == (8, 5)
    np.testing.assert_allclose(
        a.detach().numpy().sum(axis=-1), np.ones(8), atol=1e-5
    )
    a2, logp = agent.act_and_logp(obs, deterministic=False)
    assert a2.shape == (8, 5)
    assert logp.shape == (8,)
    np.testing.assert_allclose(
        a2.detach().numpy().sum(axis=-1), np.ones(8), atol=1e-5
    )
    # Non-negativity for softmax.
    assert float(a2.detach().min()) >= -1e-6


def test_ppo_synthetic_bandit_converges():
    """Known optimal one-hot weight; KL to optimum should fall."""
    torch.manual_seed(1)
    np.random.seed(1)
    k = 4
    opt = 2
    agent = PPOAgent(
        obs_dim=k,
        action_dim=k,
        lr=3e-3,
        hidden=64,
        weight_head="softmax",
        entropy_coef=0.01,
    )
    # Constant obs; reward = weight[opt].
    def collect(n: int = 128):
        obs = torch.ones(n, k)
        with torch.no_grad():
            raw, logp = agent.act_and_logp_raw(obs, deterministic=False)
            weights = agent.raw_to_weights(raw)
        rewards = weights[:, opt].detach().clone()
        next_obs = obs.clone()
        dones = torch.zeros(n)
        return obs, raw, rewards, next_obs, dones, logp

    def mean_opt_weight() -> float:
        obs = torch.ones(64, k)
        with torch.no_grad():
            a = agent.act(obs, deterministic=True)
        return float(a[:, opt].mean())

    w0 = mean_opt_weight()
    for _ in range(60):
        obs, actions, rewards, next_obs, dones, logp = collect()
        agent.train_epoch(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
            old_logprobs=logp,
            n_epochs=4,
            n_minibatches=4,
        )
    w1 = mean_opt_weight()
    assert w1 > w0 + 0.05, f"expected learning toward arm {opt}: {w0=} -> {w1=}"
    assert w1 > 0.30, f"weight on optimal arm too low: {w1}"


def test_ppo_shuffled_reward_control_no_improvement():
    """Shuffled rewards should not systematically improve optimal-arm weight."""
    torch.manual_seed(2)
    np.random.seed(2)
    k = 4
    opt = 1
    agent = PPOAgent(
        obs_dim=k,
        action_dim=k,
        lr=3e-3,
        hidden=64,
        weight_head="softmax",
        entropy_coef=0.02,
    )

    def mean_opt() -> float:
        with torch.no_grad():
            a = agent.act(torch.ones(32, k), deterministic=True)
        return float(a[:, opt].mean())

    w0 = mean_opt()
    for _ in range(25):
        n = 96
        obs = torch.ones(n, k)
        with torch.no_grad():
            raw, logp = agent.act_and_logp_raw(obs, deterministic=False)
        rewards = torch.randn(n)
        agent.train_epoch(
            obs=obs,
            actions=raw,
            rewards=rewards,
            next_obs=obs.clone(),
            dones=torch.zeros(n),
            old_logprobs=logp,
            n_epochs=2,
            n_minibatches=2,
        )
    w1 = mean_opt()
    # No strong directed learning expected.
    assert abs(w1 - w0) < 0.25, f"shuffled-reward drifted too much: {w0=} {w1=}"


def test_make_single_agent_ppo_accepts_new_kwargs():
    agent = make_single_agent(
        "ppo",
        obs_dim=3,
        action_dim=2,
        lr=1e-3,
        entropy_coef=0.02,
        weight_head="softmax",
        gamma=0.99,
        gae_lambda=0.95,
        rl_backend="custom",
    )
    assert isinstance(agent, PPOAgent)
    assert agent.entropy_coef == pytest.approx(0.02, **FLOAT_TOL)
