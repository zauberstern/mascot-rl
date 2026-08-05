"""Rebalance cadence helpers: daily / weekly / monthly / quarterly_63d masks."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def month_end_mask(dates: Sequence[pd.Timestamp] | pd.DatetimeIndex | np.ndarray) -> np.ndarray:
    """True on the last trading day of each calendar month."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    if len(idx) == 0:
        return np.zeros(0, dtype=bool)
    s = pd.Series(np.arange(len(idx)), index=idx)
    last = s.groupby(s.index.to_period("M")).transform("max")
    return (s == last).to_numpy(dtype=bool)


def week_end_mask(dates: Sequence[pd.Timestamp] | pd.DatetimeIndex | np.ndarray) -> np.ndarray:
    """True on the last trading day of each ISO week."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    if len(idx) == 0:
        return np.zeros(0, dtype=bool)
    s = pd.Series(np.arange(len(idx)), index=idx)
    last = s.groupby(s.index.to_period("W")).transform("max")
    return (s == last).to_numpy(dtype=bool)


def quarterly_63d_mask(
    dates: Sequence[pd.Timestamp] | pd.DatetimeIndex | np.ndarray,
    *,
    every: int = 63,
    anchor_index: int = 0,
) -> np.ndarray:
    """True on trading-day indices ``anchor, anchor+every, ...`` (not calendar quarters)."""
    n = len(pd.DatetimeIndex(pd.to_datetime(list(dates))))
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    step = max(int(every), 1)
    start = int(anchor_index)
    if start < 0 or start >= n:
        raise ValueError(f"anchor_index={anchor_index} out of range for n={n}")
    out[start::step] = True
    return out


def build_rebalance_mask(
    dates: Sequence[pd.Timestamp] | pd.DatetimeIndex | np.ndarray,
    cadence: str,
) -> np.ndarray | None:
    """Return a boolean rebalance mask, or None for daily (every day)."""
    key = str(cadence or "daily").lower().strip()
    if key in ("daily", "day", "1d"):
        return None
    if key in ("monthly", "month", "1m"):
        return month_end_mask(dates)
    if key in ("weekly", "week", "1w"):
        return week_end_mask(dates)
    if key in ("quarterly_63d", "quarterly", "63d", "q63"):
        return quarterly_63d_mask(dates, every=63, anchor_index=0)
    raise ValueError(
        f"unknown rebalance cadence={cadence!r}; expected daily|weekly|monthly|quarterly_63d"
    )


def build_universe_cadence_mask(
    dates: Sequence[pd.Timestamp] | pd.DatetimeIndex | np.ndarray,
    mode: str,
    *,
    anchor_index: int = 0,
) -> np.ndarray:
    """Universe reselect mask (always an array; never None).

    Distinct from the policy rebalance mask. ``quarterly_63d`` is the CRUCIBLE
    slow cadence; other modes reuse the policy helpers.
    """
    key = str(mode or "quarterly_63d").lower().strip()
    if key in ("quarterly_63d", "quarterly", "63d", "q63"):
        return quarterly_63d_mask(dates, every=63, anchor_index=anchor_index)
    if key in ("daily", "day", "1d"):
        return np.ones(len(list(dates)), dtype=bool)
    if key in ("monthly", "month", "1m"):
        return month_end_mask(dates)
    if key in ("weekly", "week", "1w"):
        return week_end_mask(dates)
    raise ValueError(f"unknown universe cadence={mode!r}")


def assert_universe_subset_of_policy(
    universe_mask: np.ndarray,
    policy_mask: np.ndarray | None,
) -> None:
    """Fail closed if a universe reselect falls on a non-policy day when policy is masked."""
    u = np.asarray(universe_mask, dtype=bool).reshape(-1)
    if policy_mask is None:
        return
    p = np.asarray(policy_mask, dtype=bool).reshape(-1)
    if u.size != p.size:
        raise ValueError(f"mask length mismatch universe={u.size} policy={p.size}")
    if not np.all(u <= p):
        raise AssertionError(
            "universe cadence mask must be a subset of the policy rebalance mask"
        )


def annualized_turnover(turnover: np.ndarray, *, periods: int = 252) -> float:
    """Annualize mean daily L1 turnover."""
    x = np.asarray(turnover, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.mean(x) * periods)


def slice_rebalance_mask(
    mask: np.ndarray | None, idx: np.ndarray
) -> np.ndarray | None:
    """Row-slice a full-panel rebalance mask to a CPCV fold index set."""
    if mask is None:
        return None
    m = np.asarray(mask, dtype=bool).reshape(-1)
    ii = np.asarray(idx, dtype=int).reshape(-1)
    if ii.size == 0:
        return np.zeros(0, dtype=bool)
    if int(ii.max()) >= m.size:
        raise ValueError(
            f"rebalance_mask length {m.size} cannot cover idx max {int(ii.max())}"
        )
    return m[ii]
