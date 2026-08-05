"""Static-arbitrage screens on the OptionMetrics surface (Gatheral–Jacquier).

Calendar: total implied variance ``w = σ²·τ`` must be non-decreasing in tenor.
Butterfly: call (put) prices must be convex in strike on each expiry slice.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# Numerical tolerance for floating mid quotes / IV.
_EPS = 1e-8


def calendar_violations(
    surface: pd.DataFrame,
    *,
    iv_col: str = "impl_volatility",
    days_col: str = "days",
    delta_col: str = "delta",
    cp_col: str = "cp_flag",
    secid_col: str = "secid",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Return ``(secid, date, n_violations)`` for calendar-spread violations.

    For each (secid, date, delta, cp_flag) slice sorted by ``days``, flag any
    adjacent decrease in total variance ``w = iv² · (days/365)``.
    """
    need = {secid_col, date_col, days_col, delta_col, cp_col, iv_col}
    missing = need - set(surface.columns)
    if missing:
        raise KeyError(f"calendar_violations missing columns {sorted(missing)}")
    if surface.empty:
        return pd.DataFrame(columns=[secid_col, date_col, "n_violations"])

    df = surface[list(need)].dropna(subset=[iv_col, days_col]).copy()
    df[days_col] = pd.to_numeric(df[days_col], errors="coerce")
    df[iv_col] = pd.to_numeric(df[iv_col], errors="coerce")
    df = df.dropna(subset=[iv_col, days_col])
    df = df[df[days_col] > 0]
    if df.empty:
        return pd.DataFrame(columns=[secid_col, date_col, "n_violations"])

    df["w"] = (df[iv_col].astype(float) ** 2) * (df[days_col].astype(float) / 365.0)
    df = df.sort_values([secid_col, date_col, delta_col, cp_col, days_col])
    gcols = [secid_col, date_col, delta_col, cp_col]
    df["w_prev"] = df.groupby(gcols, sort=False)["w"].shift(1)
    df["days_prev"] = df.groupby(gcols, sort=False)[days_col].shift(1)
    viol = df["w_prev"].notna() & (df["w"] + _EPS < df["w_prev"])
    hits = df.loc[viol, [secid_col, date_col]]
    if hits.empty:
        return pd.DataFrame(columns=[secid_col, date_col, "n_violations"])
    out = (
        hits.groupby([secid_col, date_col], as_index=False)
        .size()
        .rename(columns={"size": "n_violations"})
    )
    return out


def butterfly_violations(
    chain: pd.DataFrame,
    *,
    strike_col: str = "strike",
    mid_col: str = "mid",
    cp_col: str = "cp_flag",
    exdate_col: str = "exdate",
    secid_col: str = "secid",
    date_col: str = "date",
    eps: float = _EPS,
) -> pd.DataFrame:
    """
    Return ``(secid, date, n_violations)`` for butterfly (strike-convexity) fails.

    For ordered strikes K1 < K2 < K3 on the same (secid, date, cp_flag, exdate):

        (K3-K2)·C(K1) − (K3-K1)·C(K2) + (K2-K1)·C(K3) > −eps
    """
    need = {secid_col, date_col, cp_col, exdate_col, strike_col, mid_col}
    missing = need - set(chain.columns)
    if missing:
        raise KeyError(f"butterfly_violations missing columns {sorted(missing)}")
    if chain.empty:
        return pd.DataFrame(columns=[secid_col, date_col, "n_violations"])

    df = chain[list(need)].dropna(subset=[strike_col, mid_col]).copy()
    df[strike_col] = pd.to_numeric(df[strike_col], errors="coerce")
    df[mid_col] = pd.to_numeric(df[mid_col], errors="coerce")
    df = df.dropna(subset=[strike_col, mid_col])
    if df.empty:
        return pd.DataFrame(columns=[secid_col, date_col, "n_violations"])

    df = df.sort_values([secid_col, date_col, cp_col, exdate_col, strike_col])
    records: list[tuple] = []
    gcols = [secid_col, date_col, cp_col, exdate_col]
    for key, grp in df.groupby(gcols, sort=False):
        strikes = grp[strike_col].to_numpy(dtype=np.float64)
        mids = grp[mid_col].to_numpy(dtype=np.float64)
        # Deduplicate strikes (keep first mid).
        _, uniq_idx = np.unique(strikes, return_index=True)
        order = np.sort(uniq_idx)
        strikes = strikes[order]
        mids = mids[order]
        n = strikes.size
        if n < 3:
            continue
        n_bad = 0
        # Adjacent triples are the discrete butterfly; full O(n³) is unnecessary
        # for screening and too slow on dense strike grids.
        for i in range(n - 2):
            k1, k2, k3 = strikes[i], strikes[i + 1], strikes[i + 2]
            c1, c2, c3 = mids[i], mids[i + 1], mids[i + 2]
            convex = (k3 - k2) * c1 - (k3 - k1) * c2 + (k2 - k1) * c3
            if convex <= -eps:
                n_bad += 1
        if n_bad:
            records.append((key[0], key[1], n_bad))

    if not records:
        return pd.DataFrame(columns=[secid_col, date_col, "n_violations"])
    raw = pd.DataFrame(records, columns=[secid_col, date_col, "n_violations"])
    return (
        raw.groupby([secid_col, date_col], as_index=False)["n_violations"].sum()
    )


def violation_key_set(violations: pd.DataFrame) -> set[tuple]:
    """Normalize to a set of ``(secid, date_str)`` keys."""
    if violations is None or violations.empty:
        return set()
    out: set[tuple] = set()
    for _, row in violations.iterrows():
        d = row["date"]
        d_str = str(pd.Timestamp(d).date()) if d is not None else ""
        out.add((int(row["secid"]), d_str))
    return out


def merge_violation_keys(*frames: pd.DataFrame) -> set[tuple]:
    keys: set[tuple] = set()
    for fr in frames:
        keys |= violation_key_set(fr)
    return keys


def filter_long_marks(
    long_df: pd.DataFrame,
    bad_keys: Iterable[tuple],
) -> pd.DataFrame:
    """Drop rows whose ``(secid, date)`` is in ``bad_keys``."""
    if long_df.empty or not bad_keys:
        return long_df
    bad = set(bad_keys)
    dates = pd.to_datetime(long_df["date"]).dt.strftime("%Y-%m-%d")
    mask = [
        (int(s), d) not in bad
        for s, d in zip(long_df["secid"].tolist(), dates.tolist())
    ]
    return long_df.loc[mask].reset_index(drop=True)
