"""RC4: softmax must escape equal-weight attractor via tilt_gain / actor gain."""
from __future__ import annotations

import numpy as np
import torch


def test_tilt_gain_sharpens_softmax() -> None:
    """weight_head_tilt_gain > 1 must produce more concentrated softmax output."""
    from mascotrl.policy.single_agent import _apply_weight_head

    raw = torch.randn(1, 100)
    w_base = _apply_weight_head(raw, "softmax", tilt_gain=1.0)
    w_sharp = _apply_weight_head(raw, "softmax", tilt_gain=2.0)
    assert w_sharp.max().item() > w_base.max().item()


def test_actor_final_gain_from_config() -> None:
    """actor_final_gain=0.1 must produce larger initial logits than 0.01."""
    from mascotrl.policy.single_agent import PPOAgent

    agent_small = PPOAgent(obs_dim=20, action_dim=10, actor_final_gain=0.01)
    agent_large = PPOAgent(obs_dim=20, action_dim=10, actor_final_gain=0.1)
    obs = torch.randn(1, 20)
    with torch.no_grad():
        raw_small = agent_small.net.mean(obs) if hasattr(agent_small.net, "mean") else agent_small.net.actor(obs)
        raw_large = agent_large.net.mean(obs) if hasattr(agent_large.net, "mean") else agent_large.net.actor(obs)
    assert raw_large.abs().mean().item() > raw_small.abs().mean().item() * 3


def test_softmax_tilt_gain_changes_entropy() -> None:
    """Behavioral contract: larger tilt_gain lowers softmax entropy (sharper)."""
    from mascotrl.policy.single_agent import _apply_weight_head

    raw = torch.linspace(-1.0, 1.0, 100).unsqueeze(0)
    w1 = _apply_weight_head(raw, "softmax", tilt_gain=1.0).numpy().reshape(-1)
    w5 = _apply_weight_head(raw, "softmax", tilt_gain=5.0).numpy().reshape(-1)

    def _entropy(w: np.ndarray) -> float:
        p = np.clip(w, 1e-12, 1.0)
        return float(-(p * np.log(p)).sum())

    assert _entropy(w5) < _entropy(w1)
    assert w5.max() > w1.max()
