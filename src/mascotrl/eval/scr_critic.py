"""SCR critic mix (Scenario Context Rollout) — PPO + historical only.

Critic target: Y = (1 - beta) * Y_r + beta * Y_cf.
Actor reward is never mixed.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import torch

from mascotrl.policy.single_agent import compute_gae

SCR_MODES = frozenset({"off", "nocf", "full", "speculative"})


def resolve_scr_mix(cfg: Mapping[str, Any]) -> tuple[str, float]:
    """Return ``(scr_mix, beta)`` after RASP refusals."""
    mode = str(cfg.get("scr_mix") or "off").lower().strip()
    if mode not in SCR_MODES:
        raise ValueError(f"unknown scr_mix={mode!r}; allowed={sorted(SCR_MODES)}")
    beta = float(cfg.get("scr_beta", 0.5) if mode == "full" else 0.0)
    if mode == "full":
        algo = str(cfg.get("algo") or cfg.get("policy_algo") or "ppo").lower()
        world = str(cfg.get("train_world") or cfg.get("train_distribution") or "historical").lower()
        if algo != "ppo" or world != "historical":
            raise ValueError(
                "scr_full_requires_ppo_historical: scr_mix='full' requires "
                "algo='ppo' and train_world='historical'"
            )
    return mode, beta


def assert_psi_leak_safe(psi_indices: torch.Tensor | list[int], *, t: int) -> None:
    """Fail closed if scenario map ``psi`` touches any timestamp after ``t``."""
    if isinstance(psi_indices, torch.Tensor):
        idxs = psi_indices.detach().reshape(-1).tolist()
    else:
        idxs = list(psi_indices)
    if not idxs:
        return
    mx = max(int(i) for i in idxs)
    if mx > int(t):
        raise ValueError(
            f"scr_psi_lookahead: max psi index {mx} > t={t}; "
            "scenario map must be leak-safe"
        )


def mix_critic_targets(
    y_r: torch.Tensor,
    y_cf: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    b = float(beta)
    if b <= 0.0:
        return y_r
    if b >= 1.0:
        return y_cf
    return (1.0 - b) * y_r + b * y_cf


def build_scr_returns(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    scr_mix: str = "off",
    scr_beta: float = 0.5,
    y_cf: torch.Tensor | None = None,
    psi_indices_by_t: Callable[[int], list[int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Compute advantages/returns with optional SCR critic mix.

    ``rewards`` are the actor/env estimand and are returned unchanged in meta
    so callers can assert the actor reward was never SCR-mixed.
    """
    mode = str(scr_mix or "off").lower()
    adv_r, ret_r = compute_gae(
        rewards, values, next_values, dones, gamma=gamma, gae_lambda=gae_lambda
    )
    meta: dict[str, Any] = {
        "scr_mix": mode,
        "scr_beta": float(scr_beta) if mode == "full" else 0.0,
        "actor_reward_checksum": float(rewards.detach().sum()),
    }
    if mode in ("off", "speculative") or mode == "nocf" and y_cf is None:
        # nocf with identity psi: Y_cf := Y_r
        if mode == "nocf":
            meta["scr_beta"] = 0.0
            meta["scr_nocf_identity"] = True
        return adv_r, ret_r, meta

    if mode == "nocf":
        # Plumbing active but counterfactual equals realised.
        y_cf_use = ret_r
        beta = 0.0
    else:
        if y_cf is None:
            raise ValueError("scr_mix='full' requires y_cf counterfactual returns")
        y_cf_use = y_cf
        beta = float(scr_beta)
        if psi_indices_by_t is not None:
            for t in range(int(rewards.shape[0])):
                assert_psi_leak_safe(psi_indices_by_t(t), t=t)

    ret = mix_critic_targets(ret_r, y_cf_use, beta=beta)
    # Re-derive advantages consistent with mixed returns: A = Y - V.
    adv = ret - values
    meta["scr_beta"] = beta
    return adv, ret, meta
