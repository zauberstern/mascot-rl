"""Policy-mode overlays: mandate-style constraint/objective knobs (Part D.8).

``policy_mode`` does not change the learning algorithm. It scales turnover
caps and risk aversion, and (under crisis) tightens the Amihud screen.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

POLICY_MODE_IDS: tuple[str, ...] = (
    "shared",
    "archetype_carry",
    "archetype_inflation",
    "archetype_crisis",
)

# Turnover multiplier relative to cfg turnover_limit / tau_end.
_TURNOVER_MULT: dict[str, float] = {
    "shared": 1.0,
    "archetype_carry": 0.5,
    "archetype_inflation": 1.0,
    "archetype_crisis": 2.0,
}

# Risk-aversion multiplier applied to cao_c / kappa / lam overlays.
_RISK_AVERSION_MULT: dict[str, float] = {
    "shared": 1.0,
    "archetype_carry": 0.5,
    "archetype_inflation": 1.0,  # term-spread conditioning applied separately
    "archetype_crisis": 2.0,
}


def resolve_policy_mode(cfg: Mapping[str, Any] | None) -> str:
    raw = None
    if cfg:
        for key in ("policy_mode", "mandate", "archetype_mode"):
            if key in cfg and cfg[key] is not None and str(cfg[key]).strip():
                val = cfg[key]
                if isinstance(val, bool):
                    continue
                raw = str(val).strip()
                break
    if raw is None or not raw:
        return "shared"
    key = raw.lower()
    if key not in POLICY_MODE_IDS:
        raise ValueError(
            f"unknown policy_mode={raw!r}; allowed={list(POLICY_MODE_IDS)}"
        )
    return key


def turnover_multiplier(policy_mode: str) -> float:
    mode = resolve_policy_mode({"policy_mode": policy_mode})
    return float(_TURNOVER_MULT[mode])


def apply_turnover_multiplier(tau: float, policy_mode: str) -> float:
    return float(tau) * turnover_multiplier(policy_mode)


def risk_aversion_multiplier(
    policy_mode: str,
    *,
    term_spread_z: float | None = None,
) -> float:
    """Return the risk-aversion scale for ``policy_mode``.

    ``archetype_inflation`` optionally conditions on a causal term-spread
    z-score: higher term spread → higher risk aversion (desk inflation
    mandate). When ``term_spread_z`` is None, the base mult is 1.0.
    """
    mode = resolve_policy_mode({"policy_mode": policy_mode})
    base = float(_RISK_AVERSION_MULT[mode])
    if mode == "archetype_inflation" and term_spread_z is not None:
        # Soft scale: 1 + 0.25 * clip(z, -2, 2)
        z = max(-2.0, min(2.0, float(term_spread_z)))
        return base * (1.0 + 0.25 * z)
    return base


def apply_risk_aversion(
    base: float,
    policy_mode: str,
    *,
    term_spread_z: float | None = None,
) -> float:
    return float(base) * risk_aversion_multiplier(
        policy_mode, term_spread_z=term_spread_z
    )


def resolve_term_spread_z_for_train(
    cfg: Mapping[str, Any] | None,
    *,
    dates: Sequence | None = None,
) -> float | None:
    """Resolve a causal term-spread z for ``archetype_inflation`` training.

    Preference order:
    1. Explicit ``cfg['term_spread_z']`` / ``cfg['_term_spread_z_mean']``
    2. Mean fioracle ``term_spread_level`` z-scored over the provided dates
       (graceful None when lake/dates are unavailable).
    """
    if not cfg:
        return None
    for key in ("term_spread_z", "_term_spread_z_mean"):
        if cfg.get(key) is not None:
            return float(cfg[key])
    mode = resolve_policy_mode(cfg)
    if mode != "archetype_inflation":
        return None
    if not dates:
        return None
    try:
        from pathlib import Path

        import numpy as np
        import pandas as pd

        from src.data.fioracle_macro import (
            build_fioracle_feature_frame,
            load_fioracle_macro,
        )

        lake_root = Path(str(cfg.get("lake_root") or "lake"))
        idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
        if len(idx) == 0:
            return None
        start = (idx.min() - pd.Timedelta(days=800)).strftime("%Y-%m-%d")
        end = idx.max().strftime("%Y-%m-%d")
        levels = load_fioracle_macro(
            lake_root=lake_root, start_date=start, end_date=end
        )
        feats = build_fioracle_feature_frame(levels)
        if "term_spread_level" not in feats.columns:
            return None
        series = feats["term_spread_level"].dropna()
        if series.size < 20:
            return None
        mu = float(series.mean())
        sd = float(series.std(ddof=0))
        if sd < 1e-12:
            return None
        aligned = feats["term_spread_level"].reindex(idx).to_numpy(dtype=np.float64)
        finite = aligned[np.isfinite(aligned)]
        if finite.size == 0:
            return None
        return float((float(np.mean(finite)) - mu) / sd)
    except Exception:
        return None


def amihud_drop_pct_for_mode(policy_mode: str, *, base: float = 95.0, crisis: float = 90.0) -> float:
    mode = resolve_policy_mode({"policy_mode": policy_mode})
    if mode == "archetype_crisis":
        return float(crisis)
    return float(base)


def tighter_name_cap(policy_mode: str) -> bool:
    return resolve_policy_mode({"policy_mode": policy_mode}) == "archetype_crisis"


def per_regime_sharpe(
    returns,
    regime_labels,
    *,
    periods: int = 252,
) -> dict[str, float | int]:
    """Sharpe conditioned on calm / inflationary / crisis labels."""
    import numpy as np

    from src.data.regime_labels import REGIME_IDS

    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    labs = list(regime_labels)
    if len(labs) != r.size:
        raise ValueError(
            f"returns length {r.size} != regime_labels length {len(labs)}"
        )
    out: dict[str, float | int] = {}
    for rid in REGIME_IDS:
        mask = np.asarray([str(x) == rid for x in labs], dtype=bool)
        n = int(mask.sum())
        out[f"n_days_{rid}"] = n
        if n < 2:
            out[f"sharpe_{rid}"] = float("nan")
            continue
        sub = r[mask]
        mu = float(sub.mean())
        sd = float(sub.std(ddof=0))
        out[f"sharpe_{rid}"] = float("nan") if sd < 1e-18 else mu / sd * (float(periods) ** 0.5)
    return out
