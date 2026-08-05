"""Causal three-state macro regime labels (calm / inflationary / crisis).

Deterministic rule-based taxonomy (not a port of fioracle's L1 jump model).
Quantiles at date t use the expanding window ending at t-1 only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIME_IDS: tuple[str, ...] = ("calm", "inflationary", "crisis")


def _empirical_pct(hist: np.ndarray, value: float) -> float:
    """Fraction of hist <= value (percentile rank in [0, 1])."""
    if hist.size == 0 or not np.isfinite(value):
        return float("nan")
    return float(np.mean(hist <= value))


def _apply_persistence(raw: list[str], persistence_days: int) -> list[str]:
    """Causal minimum-run filter: forward scan, no lookahead."""
    if not raw:
        return []
    persist = max(1, int(persistence_days))
    current = raw[0]
    pending: str | None = None
    pending_count = 0
    out: list[str] = []
    for lab in raw:
        if lab == current:
            pending = None
            pending_count = 0
            out.append(current)
            continue
        if pending != lab:
            pending = lab
            pending_count = 1
        else:
            pending_count += 1
        if pending_count >= persist:
            current = pending
            pending = None
            pending_count = 0
            out.append(current)
        else:
            out.append(current)
    return out


def label_regimes(
    macro: pd.DataFrame,
    *,
    min_history_days: int = 756,
    crisis_vix_q: float = 0.85,
    crisis_oas_q: float = 0.85,
    infl_q: float = 0.70,
    persistence_days: int = 10,
) -> tuple[pd.Series, pd.DataFrame]:
    """Causal three-state regime label.

    For each date t, quantiles are computed on the expanding window ending
    at t-1, so no future information can enter a label. Before
    min_history_days of history the label is 'calm' and the row is flagged
    'warmup' in the companion frame.

    Returns
    -------
    labels
        Series of regime ids indexed like ``macro``.
    regime_meta
        Auditable companion frame with warmup, persistence, and percentile ranks.
    """
    need = ("vix_level", "hy_oas_level", "inflation_yoy_level")
    missing = [c for c in need if c not in macro.columns]
    if missing:
        raise ValueError(f"macro missing columns for regime labels: {missing}")
    df = macro.sort_index()
    n = len(df)
    vix = df["vix_level"].to_numpy(dtype=np.float64)
    oas = df["hy_oas_level"].to_numpy(dtype=np.float64)
    infl = df["inflation_yoy_level"].to_numpy(dtype=np.float64)

    raw: list[str] = []
    warmup = np.zeros(n, dtype=bool)
    vix_pct = np.full(n, np.nan)
    oas_pct = np.full(n, np.nan)
    infl_pct = np.full(n, np.nan)

    min_hist = int(min_history_days)
    for i in range(n):
        if i < min_hist:
            warmup[i] = True
            raw.append("calm")
            continue
        # Expanding window through t-1
        vix_pct[i] = _empirical_pct(vix[:i], float(vix[i]))
        oas_pct[i] = _empirical_pct(oas[:i], float(oas[i]))
        infl_pct[i] = _empirical_pct(infl[:i], float(infl[i]))

        crisis = False
        if np.isfinite(vix_pct[i]) and vix_pct[i] >= crisis_vix_q:
            crisis = True
        if np.isfinite(oas_pct[i]) and oas_pct[i] >= crisis_oas_q:
            crisis = True
        if crisis:
            raw.append("crisis")
        elif np.isfinite(infl_pct[i]) and infl_pct[i] >= infl_q:
            raw.append("inflationary")
        else:
            raw.append("calm")

    # Persistence only on post-warmup; keep warmup as calm
    sticky = list(raw)
    if n > min_hist:
        post = _apply_persistence(raw[min_hist:], persistence_days)
        sticky = raw[:min_hist] + post

    labels = pd.Series(sticky, index=df.index, name="regime")
    days_in = np.ones(n, dtype=np.int32)
    switch = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if sticky[i] == sticky[i - 1]:
            days_in[i] = days_in[i - 1] + 1
        else:
            days_in[i] = 1
            switch[i] = True

    regime_meta = pd.DataFrame(
        {
            "date": df.index,
            "regime": sticky,
            "warmup": warmup,
            "days_in_regime": days_in,
            "switch_flag": switch,
            "vix_pct": vix_pct,
            "hy_oas_pct": oas_pct,
            "inflation_pct": infl_pct,
        },
        index=df.index,
    )
    return labels, regime_meta
