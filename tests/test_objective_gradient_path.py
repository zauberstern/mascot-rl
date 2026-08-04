"""Prove primary spectrum objectives move actor params via score-function weights."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.policy.objective_factory import (
    OBJECTIVE_GRADIENT_PATH,
    episode_weights,
    objective_gradient_path_for,
)


@pytest.mark.parametrize(
    "mode",
    [
        "mean_std_cao",
        "smse",
        "rsqp",
        "cvar_ru",
        "meanvar_kolm",
        "entropic_oce",
    ],
)
def test_primary_episode_weight_moves_actor_params(mode: str) -> None:
    torch.manual_seed(0)
    policy = nn.Linear(4, 1, bias=False)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-1)

    x = torch.randn(16, 4)
    # Synthetic episode returns (higher better); mix of gains and losses.
    g = torch.tensor(
        [-0.4, -0.2, -0.1, 0.0, 0.05, 0.1, 0.15, 0.2] * 2,
        dtype=torch.float32,
    )
    assert objective_gradient_path_for(mode, True) == "episode_weight"
    assert OBJECTIVE_GRADIENT_PATH.get(mode) == "episode_weight" or (
        OBJECTIVE_GRADIENT_PATH.get(mode) is None
        and objective_gradient_path_for(mode, True) == "episode_weight"
    )

    w = episode_weights(mode, g)
    assert w.shape == g.shape
    assert torch.isfinite(w).all()

    before = policy.weight.detach().clone()
    opt.zero_grad(set_to_none=True)
    # Score-function surrogate: E[w * log π]; here log_prob ~ policy(x) for a tiny net.
    log_prob = policy(x).squeeze(-1)
    loss = (w.detach() * log_prob).mean()
    loss.backward()
    opt.step()
    after = policy.weight.detach()
    change = (after - before).abs().sum().item()
    assert change > 0.0, f"{mode}: expected param L1 change > 0, got {change}"


def test_gradient_path_stamp_helpers() -> None:
    assert objective_gradient_path_for("mtm_pnl", True) == "dense_reward"
    assert objective_gradient_path_for("mikkila_asym", False) == "dense_reward"
    assert objective_gradient_path_for("differential_sharpe", True) == "dense_reward"
    assert objective_gradient_path_for("smse", True) == "episode_weight"
    assert objective_gradient_path_for("smse", False) == "critic_only"
    assert objective_gradient_path_for("none", False) == "critic_only"
