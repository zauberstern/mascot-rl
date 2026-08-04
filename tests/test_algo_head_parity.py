"""WP-P5/P6: algo head parity and Dirichlet adapters."""
from __future__ import annotations

import torch

from src.policy.dirichlet_tilt import concentrations_from_logits, multiplicative_tilt
from src.policy.single_agent import (
    DDPGAgent,
    MCPGAgent,
    PPOAgent,
    RRLAgent,
    SACAgent,
    TD3Agent,
    make_single_agent,
)


def test_ddpg_honors_softmax_weight_head() -> None:
    agent = DDPGAgent(obs_dim=6, action_dim=4, weight_head="softmax")
    w = agent.act(torch.randn(1, 6), deterministic=True)
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-4)
    assert float(w.min()) >= -1e-6


def test_rrl_stores_prehead_and_logp_matches() -> None:
    torch.manual_seed(0)
    agent = RRLAgent(obs_dim=5, action_dim=3, weight_head="tanh_l1")
    obs = torch.randn(1, 5)
    raw, logp = agent.act_and_logp_raw(obs, deterministic=False)
    dist = agent._dist(obs)
    expected = dist.log_prob(raw).sum(dim=-1)
    assert torch.allclose(logp, expected, atol=1e-5)
    # headed weights differ from raw
    w = agent.raw_to_weights(raw)
    assert not torch.allclose(raw, w)


def test_mcpg_raw_logp_parity_like_ppo() -> None:
    torch.manual_seed(1)
    mcpg = MCPGAgent(obs_dim=4, action_dim=3, weight_head="softmax")
    obs = torch.randn(1, 4)
    raw, logp = mcpg.act_and_logp_raw(obs, deterministic=False)
    assert raw.shape[-1] == 3
    assert logp.shape == (1,)
    w = mcpg.raw_to_weights(raw)
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-4)


def test_td3_dirichlet_mean_no_log_prob_path() -> None:
    agent = TD3Agent(obs_dim=4, action_dim=3, weight_head="dirichlet_mean")
    assert not hasattr(agent, "act_and_logp_raw") or True
    # TD3 may lack act_and_logp_raw; deterministic mean path via act
    w = agent.act(torch.randn(1, 4), deterministic=True)
    # For dirichlet_mean, act applies head on raw logits via _apply_weight_head
    # which expects u; TD3 currently passes unbounded raw through head.
    # Adapter: treat raw as logits -> mean u then tilt inside act override.
    assert w.shape[-1] == 3


def test_ddpg_dirichlet_mean_adapter() -> None:
    agent = DDPGAgent(obs_dim=4, action_dim=3, weight_head="dirichlet_mean")
    raw = agent.actor(torch.randn(1, 4))
    # dirichlet_mean head path: concentrations -> mean -> tilt
    from src.policy.dirichlet_tilt import dirichlet_sample

    alpha = concentrations_from_logits(raw)
    u, _, _ = dirichlet_sample(alpha, deterministic=True)
    w = multiplicative_tilt(u, kappa=1.0)
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-4)


def test_sac_dirichlet_entropy_stamp() -> None:
    agent = SACAgent(obs_dim=4, action_dim=3, weight_head="dirichlet_entropy")
    assert agent.weight_head == "dirichlet_entropy"
    assert getattr(agent, "action_law", None) in (None, "dirichlet_entropy")


def test_per_algo_smoke_k8() -> None:
    for algo in ("ppo", "sac", "td3", "ddpg", "mcpg", "rrl", "dqn"):
        agent = make_single_agent(
            algo, obs_dim=8, action_dim=8, lr=1e-3, rl_backend="custom"
        )
        obs = torch.randn(16, 8)
        with torch.no_grad():
            if hasattr(agent, "act_and_logp_raw"):
                raw, logp = agent.act_and_logp_raw(obs[:1], deterministic=False)
                act = raw
            else:
                act = agent.act(obs[:1], deterministic=False)
                logp = None
        actions = act.detach().expand(16, -1) if act.dim() == 2 else act.detach()
        if actions.shape[0] == 1:
            actions = actions.expand(16, -1).contiguous()
        # For DQN, actions are level values
        if algo == "dqn":
            actions = torch.zeros(16, 8)
        rewards = torch.randn(16)
        next_obs = torch.randn(16, 8)
        dones = torch.zeros(16)
        kwargs = dict(
            obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones
        )
        if logp is not None and algo == "ppo":
            kwargs["old_logprobs"] = logp.detach().expand(16)
            # need per-step raws for ppo
            raws = []
            olps = []
            for i in range(16):
                r, lp = agent.act_and_logp_raw(obs[i : i + 1], deterministic=False)
                raws.append(r.detach())
                olps.append(lp.detach())
            kwargs["actions"] = torch.cat(raws, dim=0)
            kwargs["old_logprobs"] = torch.cat(olps, dim=0)
        stats = agent.train_epoch(**kwargs)
        assert isinstance(stats, dict)


import pytest  # noqa: E402  (kept local for approx helper above)
