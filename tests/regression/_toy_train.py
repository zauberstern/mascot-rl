"""Shared toy PPO train helpers for regression / golden tests."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from mascotrl.policy.single_agent import make_single_agent


def run_toy_train(
    *,
    seed: int = 42,
    steps: int = 200,
    epochs: int = 2,
    obs_dim: int = 8,
    action_dim: int = 5,
    return_stats: bool = False,
) -> Any:
    """Short custom-PPO train on synthetic batches; returns final weights or (weights, stats)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    agent = make_single_agent(
        "ppo",
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=1e-3,
        rl_backend="custom",
    )
    batch_n = min(64, max(16, steps // 2))
    last_stats: dict[str, float] = {}
    for ep in range(epochs):
        g = torch.Generator().manual_seed(seed + ep)
        obs = torch.randn(batch_n, obs_dim, generator=g)
        actions = torch.randn(batch_n, action_dim, generator=g)
        rewards = torch.randn(batch_n, generator=g)
        next_obs = torch.randn(batch_n, obs_dim, generator=g)
        dones = torch.zeros(batch_n)
        last_stats = agent.train_epoch(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
        )
    probe = torch.zeros(1, obs_dim)
    with torch.no_grad():
        w = agent.act(probe, deterministic=True).detach().cpu()
    if return_stats:
        flat = w.numpy().reshape(-1)
        ew = np.full_like(flat, 1.0 / max(len(flat), 1))
        metrics = {
            "final_mean_reward": float(
                last_stats.get("mean_reward", last_stats.get("reward_mean", 0.0))
                or 0.0
            ),
            "final_entropy": float(
                last_stats.get("entropy", last_stats.get("policy_entropy", 0.0)) or 0.0
            ),
            "weight_l1_vs_ew": float(np.abs(flat - ew).sum()),
            "max_weight": float(np.max(np.abs(flat))),
            "total_optimizer_steps": int(epochs * batch_n),
            "final_loss": float(
                last_stats.get("loss", last_stats.get("policy_loss", 0.0)) or 0.0
            ),
            "final_weights": flat,
        }
        # Prefer true entropy from act distribution if agent exposes it.
        for k, v in last_stats.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                if "entropy" in k.lower():
                    metrics["final_entropy"] = float(v)
                if "loss" in k.lower() and "actor" not in k.lower():
                    metrics["final_loss"] = float(v)
        return w, metrics
    return w
