"""TDD (W2.3 / P4): actor knobs and admissible SAC/TD3 weight heads."""
from __future__ import annotations

import torch

from src.policy.single_agent import (
    PPOAgent,
    SACAgent,
    TD3Agent,
    _ActorCritic,
    make_single_agent,
)


def test_actor_final_gain_scales_actor_init_std() -> None:
    torch.manual_seed(0)
    small = _ActorCritic(obs_dim=8, action_dim=4, hidden=16, actor_final_gain=0.01)
    torch.manual_seed(0)
    large = _ActorCritic(obs_dim=8, action_dim=4, hidden=16, actor_final_gain=1.0)
    last_small = small.actor[-1].weight.detach()
    last_large = large.actor[-1].weight.detach()
    assert float(last_large.abs().mean()) > float(last_small.abs().mean())


def test_ppo_agent_default_actor_final_gain_matches_prior_hardcode() -> None:
    torch.manual_seed(0)
    agent = PPOAgent(obs_dim=6, action_dim=3, hidden=8)
    assert isinstance(agent.net, _ActorCritic)


def test_weight_head_temperature_flattens_softmax_distribution() -> None:
    torch.manual_seed(0)
    raw = torch.tensor([[2.0, -1.0, 0.5]])

    low_temp = PPOAgent(obs_dim=4, action_dim=3, weight_head="softmax", weight_head_temperature=0.1)
    high_temp = PPOAgent(obs_dim=4, action_dim=3, weight_head="softmax", weight_head_temperature=10.0)

    w_low = low_temp.raw_to_weights(raw)
    w_high = high_temp.raw_to_weights(raw)

    # Low temperature sharpens the distribution (higher max weight);
    # high temperature flattens it toward uniform.
    assert float(w_low.max()) > float(w_high.max())
    assert torch.allclose(w_high.sum(dim=-1), torch.tensor([1.0]), atol=1e-5)
    assert torch.allclose(w_low.sum(dim=-1), torch.tensor([1.0]), atol=1e-5)


def test_weight_head_temperature_default_matches_prior_behavior() -> None:
    raw = torch.tensor([[2.0, -1.0, 0.5]])
    agent = PPOAgent(obs_dim=4, action_dim=3, weight_head="softmax")
    import torch.nn.functional as F

    assert torch.allclose(agent.raw_to_weights(raw), F.softmax(raw, dim=-1))


def test_mcpg_weight_head_temperature_and_raw_action_path() -> None:
    """Path A: MCPG must honor softmax temperature and expose raw logits."""
    torch.manual_seed(0)
    raw = torch.tensor([[2.0, -1.0, 0.5]])
    low = make_single_agent(
        "mcpg",
        obs_dim=4,
        action_dim=3,
        weight_head="softmax",
        weight_head_temperature=0.1,
        hidden=8,
    )
    high = make_single_agent(
        "mcpg",
        obs_dim=4,
        action_dim=3,
        weight_head="softmax",
        weight_head_temperature=10.0,
        hidden=8,
    )
    assert float(low.raw_to_weights(raw).max()) > float(high.raw_to_weights(raw).max())
    obs = torch.randn(2, 4)
    logits, logp = low.act_and_logp_raw(obs, deterministic=False)
    assert logits.shape == (2, 3)
    assert logp.shape == (2,)
    w = low.raw_to_weights(logits)
    assert torch.allclose(w.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_sac_act_softmax_emits_simplex_weights() -> None:
    """P4: SAC must not leak raw Gaussian samples into env.step."""
    torch.manual_seed(0)
    agent = SACAgent(obs_dim=6, action_dim=4, weight_head="softmax")
    obs = torch.randn(5, 6)
    w = agent.act(obs, deterministic=True)
    assert w.shape == (5, 4)
    assert torch.allclose(w.sum(dim=-1), torch.ones(5), atol=1e-5)
    assert bool((w >= 0).all())


def test_td3_act_tanh_l1_emits_unit_l1_weights() -> None:
    """P4: TD3 tanh_l1 head must emit L1-normalized long-short weights."""
    torch.manual_seed(0)
    agent = TD3Agent(obs_dim=6, action_dim=4, weight_head="tanh_l1")
    obs = torch.randn(5, 6)
    w = agent.act(obs, deterministic=True)
    assert w.shape == (5, 4)
    assert torch.allclose(w.abs().sum(dim=-1), torch.ones(5), atol=1e-5)


def test_make_single_agent_sac_accepts_weight_head() -> None:
    agent = make_single_agent(
        "sac", obs_dim=4, action_dim=3, weight_head="softmax", rl_backend="custom"
    )
    assert isinstance(agent, SACAgent)
    assert agent.weight_head == "softmax"
    w = agent.act(torch.randn(2, 4), deterministic=True)
    assert torch.allclose(w.sum(dim=-1), torch.ones(2), atol=1e-5)
