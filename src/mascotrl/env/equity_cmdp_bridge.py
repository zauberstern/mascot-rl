"""Equity-native bridge into CMDPEnv for Arm B (HAPPO+CMDP spine).

CMDPEnv historically expects option surface tensors. For the eq arm we
synthesize a minimal surface/spot view from the equity return panel so the
spine builders can run without resurrecting hedge-MDP engines.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch


def equity_panel_to_cmdp_tensors(
    returns: np.ndarray,
    *,
    spot0: float = 100.0,
) -> dict[str, Any]:
    """Map ``(T, K)`` equity returns to CMDP-compatible spot / dummy surfaces.

    Returns
    -------
    dict with:
      - ``spot_paths``: ``(1, K, T)`` cumulative price levels
      - ``surfaces``: ``(1, K, T, 1, 1)`` placeholder IV (= rolling 21d vol)
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError(f"returns must be (T,K), got {r.shape}")
    t, k = r.shape
    spots = np.empty((t, k), dtype=np.float64)
    spots[0] = float(spot0)
    for i in range(1, t):
        spots[i] = spots[i - 1] * (1.0 + np.nan_to_num(r[i], nan=0.0))
    # Rolling realized vol as a 1x1 "surface" placeholder (not used as ATM proxy claim).
    iv = np.zeros((t, k), dtype=np.float64)
    for i in range(t):
        lo = max(0, i - 20)
        iv[i] = np.nanstd(r[lo : i + 1], axis=0)
    iv = np.nan_to_num(iv, nan=0.2)
    spot_paths = torch.as_tensor(spots.T[None, ...], dtype=torch.float32)  # (1,K,T)
    surfaces = torch.as_tensor(
        iv.T[None, ..., None, None], dtype=torch.float32
    )  # (1,K,T,1,1)
    return {"spot_paths": spot_paths, "surfaces": surfaces, "returns": r}


def should_route_eq_via_cmdp(cfg: Mapping[str, Any]) -> bool:
    """True when YAML explicitly requests Arm B spine routing."""
    return bool(cfg.get("route_eq_via_cmdp", False))
