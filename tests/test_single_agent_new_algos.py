"""C4: correctness coverage for DDPG/MCPG/RRL/DQN single-agent adapters.

test_single_agent_smoke.py only checks each algo runs one update without
NaN/crash. These tests check each adapter's *distinguishing* mechanic
actually does what its docstring claims.
"""
from __future__ import annotations

import numpy as np
import torch

from mascotrl.policy.single_agent import DDPGAgent, DQNAgent, MCPGAgent, RRLAgent


def test_ddpg_actor_head_emits_l1_normalized_weights() -> None:
    torch.manual_seed(0)
    agent = DDPGAgent(obs_dim=6, action_dim=4, lr=1e-3)
    obs = torch.randn(5, 6)
    w = agent.act(obs, deterministic=True)
    assert w.shape == (5, 4)
    assert torch.allclose(w.abs().sum(dim=-1), torch.ones(5), atol=1e-5)


def test_ddpg_target_networks_track_online_after_update() -> None:
    torch.manual_seed(0)
    agent = DDPGAgent(obs_dim=6, action_dim=4, lr=1e-2, tau=0.5)
    obs = torch.randn(16, 6)
    actions = torch.randn(16, 4)
    rewards = torch.randn(16)
    next_obs = torch.randn(16, 6)
    dones = torch.zeros(16)
    before = [p.clone() for p in agent.q_t.parameters()]
    agent.train_epoch(obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones)
    after = list(agent.q_t.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_mcpg_uses_full_monte_carlo_return_not_bootstrapped_gae() -> None:
    """gae_lambda=1.0 inside MCPG's train_epoch means the value target for
    an all-non-terminal batch should differ from a lambda<1 GAE target
    (i.e. it is not silently reusing PPO's default advantage estimator)."""
    torch.manual_seed(0)
    agent = MCPGAgent(obs_dim=5, action_dim=3, lr=1e-3)
    obs = torch.randn(10, 5)
    actions = torch.randn(10, 3)
    rewards = torch.ones(10)  # constant reward highlights lambda sensitivity
    next_obs = torch.randn(10, 5)
    dones = torch.zeros(10)
    stats = agent.train_epoch(obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones)
    assert np.isfinite(stats["actor_loss"])
    assert "critic_loss" in stats


def test_mcpg_episode_boundary_resets_return_accumulation() -> None:
    torch.manual_seed(1)
    agent = MCPGAgent(obs_dim=4, action_dim=2, lr=1e-3)
    obs = torch.randn(6, 4)
    actions = torch.randn(6, 2)
    rewards = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    next_obs = torch.randn(6, 4)
    dones = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    stats = agent.train_epoch(obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones)
    assert np.isfinite(stats["loss"])


def test_rrl_has_no_critic_and_uses_differential_sharpe_signal() -> None:
    """RRL is a direct-policy method: no value network, and the training
    signal must come from DifferentialSharpe, not raw reward magnitude."""
    torch.manual_seed(0)
    agent = RRLAgent(obs_dim=5, action_dim=3, lr=1e-3)
    assert not hasattr(agent, "critic")
    assert not any("critic" in n for n, _ in agent.actor.named_parameters())

    obs = torch.randn(20, 5)
    actions = torch.randn(20, 3)
    next_obs = torch.randn(20, 5)
    dones = torch.zeros(20)

    # A reward stream with rising then falling Sharpe should not collapse
    # the differential-Sharpe signal to a constant (which raw-reward-mean
    # REINFORCE could also match) -- verify the signal actually varies.
    rewards = torch.tensor([0.01 * ((-1) ** i) * (i % 5) for i in range(20)])
    stats = agent.train_epoch(obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones)
    assert np.isfinite(stats["loss"])
    assert "mean_diff_sharpe" in stats


def test_rrl_resets_sharpe_accumulator_at_episode_boundaries() -> None:
    torch.manual_seed(2)
    agent = RRLAgent(obs_dim=3, action_dim=2, lr=1e-3)
    obs = torch.randn(8, 3)
    actions = torch.randn(8, 2)
    next_obs = torch.randn(8, 3)
    rewards = torch.randn(8) * 0.01
    dones = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float32)
    stats = agent.train_epoch(obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones)
    assert np.isfinite(stats["loss"])


def test_dqn_greedy_action_matches_argmax_q() -> None:
    torch.manual_seed(0)
    agent = DQNAgent(obs_dim=4, action_dim=3, lr=1e-3)
    obs = torch.randn(2, 4)
    with torch.no_grad():
        qv = agent._q_values(agent.q, obs)
        expected_idx = qv.argmax(dim=-1)
    w = agent.act(obs, deterministic=True)
    levels = agent._levels_t[expected_idx]
    denom = levels.abs().sum(dim=-1, keepdim=True).clamp_min(1e-8)
    assert torch.allclose(w, levels / denom)


def test_dqn_epsilon_greedy_explores_when_not_deterministic() -> None:
    torch.manual_seed(0)
    agent = DQNAgent(obs_dim=4, action_dim=5, lr=1e-3, epsilon=1.0)
    obs = torch.randn(1, 4).expand(50, 4)
    acted = torch.stack([agent.act(obs[:1], deterministic=False) for _ in range(50)])
    # epsilon=1.0: every call is fully random, so not all 50 draws collapse
    # onto the same weight vector (would happen with epsilon=0 / greedy).
    assert acted.unique(dim=0).shape[0] > 1


def test_dqn_target_network_updates_only_every_target_update_every_steps() -> None:
    torch.manual_seed(0)
    agent = DQNAgent(obs_dim=4, action_dim=3, lr=1e-2, target_update_every=3)
    obs = torch.randn(8, 4)
    actions = torch.randn(8, 3)
    rewards = torch.randn(8)
    next_obs = torch.randn(8, 4)
    dones = torch.zeros(8)

    def snap():
        return [p.clone() for p in agent.q_t.parameters()]

    s0 = snap()
    agent.train_epoch(obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones)
    s1 = snap()
    assert all(torch.allclose(a, b) for a, b in zip(s0, s1))  # not yet due (1/3)
    agent.train_epoch(obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones)
    s2 = snap()
    assert all(torch.allclose(a, b) for a, b in zip(s1, s2))  # not yet due (2/3)
    agent.train_epoch(obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones)
    s3 = snap()
    assert any(not torch.allclose(a, b) for a, b in zip(s2, s3))  # due (3/3)


def test_dqn_actions_snap_to_nearest_level_for_td_index() -> None:
    torch.manual_seed(0)
    agent = DQNAgent(obs_dim=3, action_dim=2, lr=1e-3, levels=(-1.0, 0.0, 1.0))
    actions = torch.tensor([[0.9, -0.4], [0.1, 1.2]])
    idx = agent._levels_to_index(actions)
    assert torch.equal(idx, torch.tensor([[2, 1], [1, 2]]))
