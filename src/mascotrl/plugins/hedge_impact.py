"""Hedge-leg impact measurement (Bouchaud-style square-root metaorder proxy).

OOS / CPCV measurement only — never soft-bake into overnight train R_t.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def sqrt_metaorder_shortfall(
    participation: float | np.ndarray,
    sigma: float | np.ndarray,
    *,
    coef: float = 1.0,
) -> float | np.ndarray:
    """I ≈ coef * σ * sqrt(|participation|); participation = Q / ADV in [0, 1+]."""
    p = np.asarray(participation, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    out = float(coef) * s * np.sqrt(np.abs(p))
    if np.ndim(out) == 0:
        return float(out)
    return out


def hedge_impact_breakdown(
    hedge_notional: float,
    adv: float,
    sigma: float,
    *,
    coef: float = 1.0,
    enabled: bool = True,
) -> dict[str, Any]:
    """Return measurement dict for cost ladder / OOS friction."""
    if not enabled or adv <= 0 or not np.isfinite(adv):
        return {
            "enabled": bool(enabled),
            "participation": 0.0,
            "sigma": float(sigma),
            "adv": float(adv) if np.isfinite(adv) else float("nan"),
            "shortfall": 0.0,
            "model": "sqrt_metaorder",
        }
    part = abs(float(hedge_notional)) / float(adv)
    short = float(sqrt_metaorder_shortfall(part, sigma, coef=coef))
    return {
        "enabled": True,
        "participation": float(part),
        "sigma": float(sigma),
        "adv": float(adv),
        "shortfall": short,
        "model": "sqrt_metaorder",
        "coef": float(coef),
    }


def friction_config_hash(cfg: Mapping[str, Any]) -> str:
    """Stable hash of claim-path friction knobs."""
    import hashlib
    import json

    keys = (
        "om_touch_enabled",
        "om_touch_fee_bps",
        "om_touch_spread_multiplier",
        "hedge_leg_spread_bps",
        "execution_spread_bps",
        "execution_impact_coef",
        "hedge_impact_coef",
        "hedge_impact_enabled",
        "funding_enabled",
        "borrow_floor_bps_annual",
        "proportional_cost",
        "kappa",
        "use_om_borrow",
    )
    payload = {k: cfg.get(k) for k in keys}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
