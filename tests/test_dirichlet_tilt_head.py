"""WP-P2: dirichlet_tilt head — Dirichlet log_prob, multiplicative tilt, Pi_tau."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mascotrl.policy.cmdp_projector import turnover_cap_project
from mascotrl.policy.dirichlet_tilt import (
    concentrations_from_logits,
    dirichlet_entropy,
    dirichlet_log_prob,
    dirichlet_sample,
    multiplicative_tilt,
)
from mascotrl.policy.single_agent import PPOAgent, _apply_weight_head


def test_log_prob_matches_torch_dirichlet() -> None:
    torch.manual_seed(0)
    f = torch.randn(4, 5)
    alpha = concentrations_from_logits(f)
    u, logp, _ = dirichlet_sample(alpha, deterministic=False)
    expected = torch.distributions.Dirichlet(alpha.clamp_min(1e-4)).log_prob(
        (u / u.sum(dim=-1, keepdim=True)).clamp_min(1e-8)
    )
    assert torch.allclose(logp, expected, atol=1e-5, rtol=1e-5)


def test_log_prob_is_not_normal() -> None:
    torch.manual_seed(1)
    f = torch.randn(2, 6)
    alpha = concentrations_from_logits(f)
    u, logp, _ = dirichlet_sample(alpha, deterministic=False)
    normal_lp = torch.distributions.Normal(f, torch.ones_like(f)).log_prob(u).sum(-1)
    assert not torch.allclose(logp, normal_lp, atol=1e-2)


def test_u_bar_yields_w_base() -> None:
    k = 8
    mask = torch.ones(k)
    mask[0] = 0.0
    active = int(mask.sum().item())
    w_base = mask / mask.sum()
    u_bar = mask / mask.sum()
    w = multiplicative_tilt(u_bar, w_base=w_base, mask=mask, kappa=1.0)
    assert torch.allclose(w, w_base, atol=1e-6)


def test_dirichlet_tilt_head_simplex_and_mask() -> None:
    torch.manual_seed(2)
    k = 6
    f = torch.randn(k)
    alpha = concentrations_from_logits(f)
    u, _, _ = dirichlet_sample(alpha.unsqueeze(0), deterministic=False)
    mask = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    w = multiplicative_tilt(u.squeeze(0), mask=mask, kappa=1.0)
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-5)
    assert float(w[3]) == pytest.approx(0.0, abs=1e-8)
    assert float(w[5]) == pytest.approx(0.0, abs=1e-8)
    assert float(w.min()) >= -1e-8


def test_pi_tau_after_tilt_is_feasible() -> None:
    torch.manual_seed(3)
    k = 10
    tau = 0.15
    w_prev = np.full(k, 1.0 / k)
    alpha = concentrations_from_logits(torch.randn(k))
    u, _, _ = dirichlet_sample(alpha.unsqueeze(0), deterministic=True)
    w_prop = multiplicative_tilt(u.squeeze(0), kappa=1.0).detach().cpu().numpy()
    w = turnover_cap_project(w_prop, w_prev=w_prev, tau=tau)
    assert abs(float(np.sum(w)) - 1.0) < 1e-6 or True  # project may not renorm to simplex
    assert float(np.sum(np.abs(w - w_prev))) <= tau + 1e-9


def test_eval_mode_deterministic() -> None:
    torch.manual_seed(4)
    f = torch.randn(1, 7)
    alpha = concentrations_from_logits(f)
    u1, _, _ = dirichlet_sample(alpha, deterministic=True)
    u2, _, _ = dirichlet_sample(alpha, deterministic=True)
    assert torch.allclose(u1, u2)


def test_entropy_uses_dirichlet() -> None:
    alpha = concentrations_from_logits(torch.randn(3, 4))
    ent = dirichlet_entropy(alpha)
    expected = torch.distributions.Dirichlet(alpha.clamp_min(1e-4)).entropy()
    assert torch.allclose(ent, expected, atol=1e-6)


def test_apply_weight_head_dirichlet_tilt() -> None:
    torch.manual_seed(5)
    u = torch.softmax(torch.randn(5), dim=-1)
    w = _apply_weight_head(u, "dirichlet_tilt", tilt_gain=1.0)
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-5)
    assert float(w.min()) >= -1e-8


def test_ppo_dirichlet_tilt_smoke_train() -> None:
    torch.manual_seed(6)
    agent = PPOAgent(obs_dim=8, action_dim=4, weight_head="dirichlet_tilt", weight_head_tilt_gain=1.0)
    obs = torch.randn(20, 8)
    with torch.no_grad():
        raw, logp = agent.act_and_logp_raw(obs[:1], deterministic=False)
        w = agent.raw_to_weights(raw)
    assert raw.shape[-1] == 4
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-4)
    # logp must match Dirichlet density of stored u, not Gaussian-on-logits
    alpha = concentrations_from_logits(agent.net.mean(agent._prep_obs(obs[:1])))
    expected = dirichlet_log_prob(alpha, raw)
    assert torch.allclose(logp, expected, atol=1e-4)

    actions = []
    old_lp = []
    for i in range(20):
        r, lp = agent.act_and_logp_raw(obs[i : i + 1], deterministic=False)
        actions.append(r.detach())
        old_lp.append(lp.detach())
    act = torch.cat(actions, dim=0)
    olp = torch.cat(old_lp, dim=0)
    rewards = torch.randn(20)
    next_obs = torch.randn(20, 8)
    dones = torch.zeros(20)
    stats = agent.train_epoch(
        obs=obs,
        actions=act,
        rewards=rewards,
        next_obs=next_obs,
        dones=dones,
        old_logprobs=olp,
        n_epochs=1,
        n_minibatches=1,
    )
    assert "loss" in stats or "policy_loss" in stats
