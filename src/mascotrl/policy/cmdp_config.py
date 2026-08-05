"""CMDP / PID-Lagrangian configuration helpers."""
from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F


def resolve_cmdp_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    block = dict(cfg.get("cmdp") or {})
    enabled = bool(block.get("enabled", False))
    return {
        "cmdp_enabled": enabled,
        "cmdp_limit_d": float(block.get("target", block.get("limit_d", 0.0)) or 0.0),
        "cmdp_cost_signal": str(block.get("cost_signal", "turnover")).lower(),
        "cmdp_alpha": float(block.get("alpha", 0.95)),
        "cmdp_kp": float(block.get("kp", 0.0)),
        "cmdp_ki": float(block.get("ki", 1e-3)),
        "cmdp_kd": float(block.get("kd", 0.0)),
    }


def build_step_costs(
    rewards: torch.Tensor,
    *,
    signal: str,
    deltas: torch.Tensor | None = None,
    alpha: float = 0.95,
) -> torch.Tensor:
    """Per-step CMDP cost tensor aligned with ``rewards``."""
    sig = str(signal).lower()
    if sig == "cvar":
        return F.relu(-rewards.reshape(-1))
    if sig == "delta" and deltas is not None:
        return deltas.abs().sum(dim=-1).reshape(-1)
    if sig == "turnover" and deltas is not None:
        return deltas.abs().sum(dim=-1).reshape(-1)
    return F.relu(-rewards.reshape(-1))
