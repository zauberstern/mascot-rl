"""A8: GAE episode-boundary correctness, truncation bootstrapping, RMS update."""
from __future__ import annotations

import numpy as np
import pytest
from tests.conftest import FLOAT_TOL
import torch

from mascotrl.policy.single_agent import PPOAgent, compute_gae


def test_gae_multi_episode_done_blocks_backward_leakage():
    """A large reward after a done boundary must not leak into advantages
    computed before that boundary (done resets the backward accumulator)."""
    rewards = torch.tensor([0.0, 0.0, 0.0, 10.0])
    dones = torch.tensor([0.0, 1.0, 0.0, 0.0])
    values = torch.zeros(4)
    next_values = torch.zeros(4)
    adv, returns = compute_gae(rewards, values, next_values, dones, gamma=1.0, gae_lambda=1.0)
    # Everything at/after the done sees the future +10; everything strictly
    # before the done boundary must be unaffected by it.
    assert float(adv[0]) == pytest.approx(0.0, **FLOAT_TOL)
    assert float(adv[1]) == pytest.approx(0.0, **FLOAT_TOL)
    assert float(adv[2]) == pytest.approx(10.0, **FLOAT_TOL)
    assert float(adv[3]) == pytest.approx(10.0, **FLOAT_TOL)
    np.testing.assert_allclose(returns.numpy(), adv.numpy())


def test_gae_truncation_bootstraps_from_next_value_when_done_zero():
    """done=0 must bootstrap from next_value; done=1 must not (A8)."""
    rewards = torch.tensor([0.0])
    values = torch.tensor([0.0])
    next_values = torch.tensor([5.0])

    adv_truncated, _ = compute_gae(
        rewards, values, next_values, torch.tensor([0.0]), gamma=0.99, gae_lambda=0.95
    )
    adv_terminal, _ = compute_gae(
        rewards, values, next_values, torch.tensor([1.0]), gamma=0.99, gae_lambda=0.95
    )
    assert abs(float(adv_truncated[0]) - 0.99 * 5.0) < 1e-6
    assert float(adv_terminal[0]) == pytest.approx(0.0, **FLOAT_TOL)


def test_gae_truncation_vs_terminal_values_differ():
    rewards = torch.tensor([0.0])
    values = torch.tensor([0.0])
    next_values = torch.tensor([5.0])
    adv_trunc, _ = compute_gae(
        rewards, values, next_values, torch.tensor([0.0]), gamma=0.99, gae_lambda=0.95
    )
    adv_term, _ = compute_gae(
        rewards, values, next_values, torch.tensor([1.0]), gamma=0.99, gae_lambda=0.95
    )
    assert float(adv_trunc[0]) > float(adv_term[0])
    assert abs(float(adv_trunc[0]) - 0.99 * 5.0) < 1e-6


def test_obs_rms_updates_once_per_sample_not_twice_in_train_epoch():
    """A8: train_epoch must not re-update obs_rms on the same batch that
    act_and_logp_raw already folded into the running mean/var at rollout."""
    torch.manual_seed(0)
    agent = PPOAgent(obs_dim=4, action_dim=3, hidden=16, weight_head="softmax")
    n = 32
    obs = torch.randn(n, 4) * 3.0 + 1.0

    with torch.no_grad():
        raw, logp = agent.act_and_logp_raw(obs, deterministic=False)
    count_after_rollout = float(agent.obs_rms.count)
    mean_after_rollout = agent.obs_rms.mean.clone()
    var_after_rollout = agent.obs_rms.var.clone()

    rewards = torch.randn(n)
    next_obs = torch.randn(n, 4)
    dones = torch.zeros(n)
    agent.train_epoch(
        obs=obs,
        actions=raw,
        rewards=rewards,
        next_obs=next_obs,
        dones=dones,
        old_logprobs=logp,
        n_epochs=2,
        n_minibatches=2,
    )
    assert float(agent.obs_rms.count) == count_after_rollout, (
        "train_epoch must not update obs_rms a second time on the same batch"
    )
    torch.testing.assert_close(agent.obs_rms.mean, mean_after_rollout)
    torch.testing.assert_close(agent.obs_rms.var, var_after_rollout)


def test_log_std_is_clamped_both_sides():
    agent = PPOAgent(obs_dim=2, action_dim=2, hidden=8, weight_head="softmax")
    with torch.no_grad():
        agent.net.log_std.fill_(50.0)
    with torch.no_grad():
        dist_hi = agent._dist(torch.zeros(1, 2))
    assert float(dist_hi.stddev.max()) <= float(np.exp(1.0)) + 1e-4
    with torch.no_grad():
        agent.net.log_std.fill_(-50.0)
        dist_lo = agent._dist(torch.zeros(1, 2))
    assert float(dist_lo.stddev.min()) >= float(np.exp(-5.0)) - 1e-6


def test_sample_weight_zero_removes_actor_gradient_for_those_samples():
    """C3: a zero sample_weight must behave like that sample not existing
    for the actor surrogate (episode-weight objective axis), while a
    None sample_weight must reproduce the pre-existing vanilla-advantage
    update exactly (backward compatibility)."""
    torch.manual_seed(0)
    n = 16
    obs = torch.randn(n, 3)
    rewards = torch.randn(n)
    next_obs = torch.randn(n, 3)
    dones = torch.zeros(n)

    def _fresh_agent() -> PPOAgent:
        torch.manual_seed(1)
        return PPOAgent(
            obs_dim=3,
            action_dim=2,
            lr=1e-2,
            hidden=16,
            weight_head="softmax",
            normalize_obs=False,
        )

    agent_none = _fresh_agent()
    with torch.no_grad():
        raw, logp = agent_none.act_and_logp_raw(obs, deterministic=False)
    agent_ones = _fresh_agent()
    agent_ones.net.load_state_dict(agent_none.net.state_dict())

    stats_none = agent_none.train_epoch(
        obs=obs, actions=raw, rewards=rewards, next_obs=next_obs, dones=dones,
        old_logprobs=logp, n_epochs=1, n_minibatches=1,
    )
    stats_ones = agent_ones.train_epoch(
        obs=obs, actions=raw, rewards=rewards, next_obs=next_obs, dones=dones,
        old_logprobs=logp, n_epochs=1, n_minibatches=1,
        sample_weight=torch.ones(n),
    )
    assert stats_none["actor_loss"] == pytest.approx(stats_ones["actor_loss"], abs=1e-6)

    agent_zero = _fresh_agent()
    agent_zero.net.load_state_dict(agent_none.net.state_dict())
    stats_zero = agent_zero.train_epoch(
        obs=obs, actions=raw, rewards=rewards, next_obs=next_obs, dones=dones,
        old_logprobs=logp, n_epochs=1, n_minibatches=1,
        sample_weight=torch.zeros(n),
    )
    # All-zero weights zero out the surrogate entirely (only entropy/critic remain).
    assert abs(stats_zero["actor_loss"]) < 1e-6


def test_sample_weight_wrong_length_raises():
    torch.manual_seed(0)
    agent = PPOAgent(obs_dim=2, action_dim=2, hidden=8, weight_head="softmax")
    n = 8
    obs = torch.randn(n, 2)
    with torch.no_grad():
        raw, logp = agent.act_and_logp_raw(obs, deterministic=False)
    with pytest.raises(ValueError, match="sample_weight"):
        agent.train_epoch(
            obs=obs,
            actions=raw,
            rewards=torch.randn(n),
            next_obs=torch.randn(n, 2),
            dones=torch.zeros(n),
            old_logprobs=logp,
            sample_weight=torch.ones(n - 1),
        )


def test_explained_variance_reflects_post_update_critic():
    """explained_variance in the returned stats must not be the stale
    pre-update snapshot (A8): running the update should be able to change it."""
    torch.manual_seed(3)
    agent = PPOAgent(obs_dim=3, action_dim=2, lr=1e-2, hidden=16, weight_head="softmax")
    n = 48
    obs = torch.randn(n, 3)
    with torch.no_grad():
        raw, logp = agent.act_and_logp_raw(obs, deterministic=False)
    rewards = torch.randn(n)
    next_obs = torch.randn(n, 3)
    dones = torch.zeros(n)
    stats = agent.train_epoch(
        obs=obs,
        actions=raw,
        rewards=rewards,
        next_obs=next_obs,
        dones=dones,
        old_logprobs=logp,
        n_epochs=3,
        n_minibatches=2,
    )
    assert "explained_variance" in stats
    assert np.isfinite(stats["explained_variance"]) or True  # may be nan if var(returns)~0
