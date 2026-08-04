"""Calendar scaling for realized return grids (honest annualization).

Hardcoding ``periods=252`` silently overstates Sharpe when the eval panel is
punctured (``.dropna(how='any')`` drops incomplete dates). Derive the annual
scale from the realized date grid instead.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def periods_per_year_from_dates(dates: Sequence[Any]) -> float:
    """Return observations-per-year implied by ``dates``.

    Uses ``(n - 1) / (calendar_span_days / 365.25)``. Raises if fewer than 30
    dates so short smoke grids do not invent a scale.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    if len(idx) < 30:
        raise ValueError(
            f"periods_per_year_from_dates requires at least 30 dates; got {len(idx)}"
        )
    span_days = float((idx[-1] - idx[0]).days)
    if span_days <= 0:
        raise ValueError("periods_per_year_from_dates requires a positive calendar span")
    return float((len(idx) - 1) / (span_days / 365.25))


def eval_panel_meta(
    dates: Sequence[Any],
    *,
    k: int,
    intended_start: str,
    intended_end: str,
    rebalance_days: int | None = None,
) -> dict[str, Any]:
    """Stamp realized vs intended eval-window coverage for the campaign artifact."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    t = int(len(idx))
    ppy = periods_per_year_from_dates(idx) if t >= 30 else float("nan")
    intended = pd.bdate_range(intended_start, intended_end)
    n_intended = int(len(intended))
    coverage = float(t / n_intended) if n_intended > 0 else float("nan")
    span_years = (
        float((idx[-1] - idx[0]).days / 365.25) if t >= 2 else float("nan")
    )
    return {
        "t": t,
        "k": int(k),
        "date_start": str(idx[0].date()) if t else None,
        "date_end": str(idx[-1].date()) if t else None,
        "calendar_span_years": span_years,
        "periods_per_year": float(ppy) if np.isfinite(ppy) else None,
        "intended_start": str(intended_start),
        "intended_end": str(intended_end),
        "business_days_intended": n_intended,
        "coverage_frac": coverage,
        "rebalance_days": int(rebalance_days) if rebalance_days is not None else None,
    }
