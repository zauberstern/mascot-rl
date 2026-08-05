"""GAE must mask on terminated only; truncations bootstrap."""
from __future__ import annotations

import torch

from mascotrl.policy.single_agent import compute_gae


def test_gae_bootstraps_on_truncation_not_terminal():
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([0.5, 0.5, 0.5])
    next_values = torch.tensor([0.6, 0.6, 0.0])
    # Truncation at t=1 (done=0) should bootstrap; termination at t=2 (done=1) cuts off.
    dones = torch.tensor([0.0, 0.0, 1.0])
    adv, _ret = compute_gae(
        rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95
    )
    assert adv[0].item() > adv[2].item()


def test_gae_truncation_vs_termination_differ_at_last_step():
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([0.1, 0.2, 0.3])
    next_values = torch.tensor([0.2, 0.3, 0.4])
    adv_trunc, _ = compute_gae(
        rewards,
        values,
        next_values,
        torch.tensor([0.0, 0.0, 0.0]),
        gamma=0.99,
        gae_lambda=0.95,
    )
    adv_term, _ = compute_gae(
        rewards,
        values,
        next_values,
        torch.tensor([0.0, 1.0, 0.0]),
        gamma=0.99,
        gae_lambda=0.95,
    )
    assert adv_trunc[1].item() > adv_term[1].item()
