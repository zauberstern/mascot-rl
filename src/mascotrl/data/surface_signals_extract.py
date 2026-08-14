"""IV surface grid extraction and per-group signal construction."""
from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from mascotrl.data.implied_moments import compute_mf_moments
from mascotrl.data.surface_signals_grid import (
    CW_DAYS,
    CW_DELTAS,
    GRID_POINTS_PER_DAY,
    KELLY_DELTAS_CALL,
    KELLY_DELTAS_PUT,
    KELLY_TENORS,
    SURFACE_SIGNAL_NAMES,
    _NAN,
)

_LOG = logging.getLogger(__name__)

# A10: BKM moment integration fails occasionally on sparse/degenerate OTM
# slices (per secid-date group). Recording every group's raise would be
# fatal for a pipeline touching hundreds of thousands of groups, so
# failures are counted and the last reason kept for post-run inspection
# instead of being silently discarded.
_BKM_MOMENT_FAILURES: dict[str, Any] = {"count": 0, "last_reason": None}


def bkm_moment_failure_count() -> int:
    """Number of ``_mf_moments_at_days`` calls that fell back to NaN."""
    return int(_BKM_MOMENT_FAILURES["count"])


def bkm_moment_last_failure_reason() -> str | None:
    return _BKM_MOMENT_FAILURES["last_reason"]


def reset_bkm_moment_failure_counter() -> None:
    _BKM_MOMENT_FAILURES["count"] = 0
    _BKM_MOMENT_FAILURES["last_reason"] = None

def _as_cp(x: Any) -> str:
    s = str(x).strip().upper()
    if s.startswith("C"):
        return "C"
    if s.startswith("P"):
        return "P"
    return s[:1]


def _to_frame(iv_panel: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if isinstance(iv_panel, pd.DataFrame):
        return iv_panel
    if isinstance(iv_panel, Mapping):
        # Flat dict of arrays, or a single-point dict.
        try:
            return pd.DataFrame(iv_panel)
        except ValueError:
            return pd.DataFrame([iv_panel])
    return pd.DataFrame(list(iv_panel))


def extract_grid_point(
    iv_panel: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    days: int,
    delta: int,
    cp_flag: str,
) -> float:
    """Return IV at a single (days, delta, cp_flag) grid node, else NaN."""
    df = _to_frame(iv_panel)
    if df.empty or "impl_volatility" not in df.columns:
        return _NAN
    want_cp = _as_cp(cp_flag)
    days_i = int(days)
    delta_i = int(delta)
    mask = (
        (pd.to_numeric(df["days"], errors="coerce") == days_i)
        & (pd.to_numeric(df["delta"], errors="coerce") == delta_i)
        & (df["cp_flag"].map(_as_cp) == want_cp)
    )
    sub = df.loc[mask, "impl_volatility"]
    if sub.empty:
        return _NAN
    val = float(pd.to_numeric(sub.iloc[0], errors="coerce"))
    if not np.isfinite(val) or val <= 0.0:
        return _NAN
    return val


def _nearest_call_iv(g: pd.DataFrame, days: int, target_delta: int) -> float:
    """Nearest available call IV at ``days`` to ``target_delta`` (absolute)."""
    exact = extract_grid_point(g, days=days, delta=target_delta, cp_flag="C")
    if np.isfinite(exact):
        return exact
    if g.empty:
        return _NAN
    calls = g[
        (pd.to_numeric(g["days"], errors="coerce") == int(days))
        & (g["cp_flag"].map(_as_cp) == "C")
    ].copy()
    if calls.empty:
        return _NAN
    calls["delta"] = pd.to_numeric(calls["delta"], errors="coerce")
    calls["impl_volatility"] = pd.to_numeric(calls["impl_volatility"], errors="coerce")
    calls = calls[np.isfinite(calls["delta"]) & np.isfinite(calls["impl_volatility"])]
    calls = calls[calls["impl_volatility"] > 0]
    if calls.empty:
        return _NAN
    dist = (calls["delta"] - float(target_delta)).abs()
    idx = dist.idxmin()
    return float(calls.loc[idx, "impl_volatility"])


def _iv_skew_30d(g: pd.DataFrame) -> float:
    put = extract_grid_point(g, days=30, delta=-20, cp_flag="P")
    call = extract_grid_point(g, days=30, delta=50, cp_flag="C")
    if not (np.isfinite(put) and np.isfinite(call)):
        return _NAN
    return float(put - call)


def _iv_term_slope(g: pd.DataFrame) -> float:
    long = extract_grid_point(g, days=365, delta=50, cp_flag="C")
    short = extract_grid_point(g, days=30, delta=50, cp_flag="C")
    if not (np.isfinite(long) and np.isfinite(short)):
        return _NAN
    return float(long - short)


def _iv_convexity_30d(g: pd.DataFrame) -> float:
    wing_lo = _nearest_call_iv(g, 30, 25)
    if not np.isfinite(wing_lo):
        wing_lo = _nearest_call_iv(g, 30, 20)
    wing_hi = _nearest_call_iv(g, 30, 75)
    if not np.isfinite(wing_hi):
        wing_hi = _nearest_call_iv(g, 30, 80)
    atm = extract_grid_point(g, days=30, delta=50, cp_flag="C")
    if not (np.isfinite(wing_lo) and np.isfinite(wing_hi) and np.isfinite(atm)):
        return _NAN
    return float(wing_lo + wing_hi - 2.0 * atm)


def _cw_vol_spread(g: pd.DataFrame) -> float:
    spreads: list[float] = []
    for days in CW_DAYS:
        for d in CW_DELTAS:
            iv_c = extract_grid_point(g, days=days, delta=d, cp_flag="C")
            iv_p = extract_grid_point(g, days=days, delta=-d, cp_flag="P")
            if np.isfinite(iv_c) and np.isfinite(iv_p):
                spreads.append(float(iv_c - iv_p))
    if not spreads:
        return _NAN
    return float(np.mean(spreads))


def _atm_iv_30(g: pd.DataFrame) -> float:
    c = extract_grid_point(g, days=30, delta=50, cp_flag="C")
    p = extract_grid_point(g, days=30, delta=-50, cp_flag="P")
    vals = [v for v in (c, p) if np.isfinite(v)]
    if not vals:
        return _NAN
    return float(np.mean(vals))


def _infer_spot(g: pd.DataFrame) -> float:
    atm = g[
        (pd.to_numeric(g["days"], errors="coerce") == 30)
        & (pd.to_numeric(g["delta"], errors="coerce").abs() == 50)
        & np.isfinite(pd.to_numeric(g["impl_strike"], errors="coerce"))
    ]
    if not atm.empty:
        return float(np.nanmedian(pd.to_numeric(atm["impl_strike"], errors="coerce")))
    strikes = pd.to_numeric(g.get("impl_strike"), errors="coerce")
    if strikes is None or not np.isfinite(strikes).any():
        return _NAN
    return float(np.nanmedian(strikes))


def _mf_moments_at_days(g: pd.DataFrame, days: int, *, rate: float = 0.0) -> dict[str, float]:
    """BKM moments from surface impl_strike / impl_premium at a fixed tenor."""
    nan = {"mfiv": _NAN, "mfis": _NAN, "mfik": _NAN}
    sub = g[pd.to_numeric(g["days"], errors="coerce") == int(days)].copy()
    if sub.empty or "impl_strike" not in sub.columns or "impl_premium" not in sub.columns:
        return nan
    sub["strike"] = pd.to_numeric(sub["impl_strike"], errors="coerce")
    sub["mid"] = pd.to_numeric(sub["impl_premium"], errors="coerce")
    ok = np.isfinite(sub["strike"]) & np.isfinite(sub["mid"]) & (sub["strike"] > 0) & (sub["mid"] >= 0)
    sub = sub.loc[ok]
    if len(sub) < 4:
        return nan
    spot = _infer_spot(g)
    if not np.isfinite(spot) or spot <= 0:
        return nan
    slice_df = pd.DataFrame(
        {
            "strike": sub["strike"].to_numpy(dtype=float),
            "mid": sub["mid"].to_numpy(dtype=float),
            "cp_flag": sub["cp_flag"].map(_as_cp).to_numpy(),
            "spot": spot,
            "rate": float(rate),
            "tau": float(days) / 365.0,
        }
    )
    try:
        out = compute_mf_moments(slice_df)
    except Exception as e:
        # A10: record the reason instead of a bare silent NaN fallback.
        _BKM_MOMENT_FAILURES["count"] += 1
        _BKM_MOMENT_FAILURES["last_reason"] = str(e)[:300]
        _LOG.debug("BKM moment integration failed at days=%s: %s", days, e)
        return nan
    return {
        "mfiv": float(out.get("mfiv", _NAN)),
        "mfis": float(out.get("mfis", _NAN)),
        "mfik": float(out.get("mfik", _NAN)),
    }


def _svix2_at_days(g: pd.DataFrame, days: int) -> float:
    """Approximate risk-neutral variance: 2 Σ premium ΔK / K² on sorted OTM strikes.

    Uses puts with K < spot and calls with K > spot. Requires ≥5 OTM points.
    """
    sub = g[pd.to_numeric(g["days"], errors="coerce") == int(days)].copy()
    if sub.empty:
        return _NAN
    spot = _infer_spot(g)
    if not np.isfinite(spot) or spot <= 0:
        return _NAN
    sub["K"] = pd.to_numeric(sub["impl_strike"], errors="coerce")
    sub["prem"] = pd.to_numeric(sub["impl_premium"], errors="coerce")
    sub["cp"] = sub["cp_flag"].map(_as_cp)
    ok = np.isfinite(sub["K"]) & np.isfinite(sub["prem"]) & (sub["K"] > 0) & (sub["prem"] >= 0)
    sub = sub.loc[ok]
    puts = sub[(sub["cp"] == "P") & (sub["K"] < spot)][["K", "prem"]]
    calls = sub[(sub["cp"] == "C") & (sub["K"] > spot)][["K", "prem"]]
    otm = pd.concat([puts, calls], ignore_index=True)
    if len(otm) < 5:
        return _NAN
    otm = otm.sort_values("K")
    k = otm["K"].to_numpy(dtype=float)
    p = otm["prem"].to_numpy(dtype=float)
    # Trapezoidal ∫ 2 Q(K)/K² dK
    w = 2.0 / (k * k)
    integ = float(np.trapezoid(w * p, k))
    if not np.isfinite(integ) or integ < 0:
        return _NAN
    return integ


def _surface_dispersion(g: pd.DataFrame) -> float:
    if "dispersion" not in g.columns:
        return _NAN
    d = pd.to_numeric(g["dispersion"], errors="coerce")
    iv = pd.to_numeric(g["impl_volatility"], errors="coerce")
    ok = np.isfinite(d) & np.isfinite(iv) & (iv > 0)
    if not ok.any():
        return _NAN
    return float(np.nanmean(d[ok]))


def _surface_quality(g: pd.DataFrame) -> float:
    iv = pd.to_numeric(g["impl_volatility"], errors="coerce")
    n_valid = int((np.isfinite(iv) & (iv > 0)).sum())
    return float(n_valid) / float(GRID_POINTS_PER_DAY)


def _signals_for_group(
    g: pd.DataFrame,
    *,
    hv: float | None = None,
    option_volume: float | None = None,
    equity_volume: float | None = None,
    borrow: float | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {name: _NAN for name in SURFACE_SIGNAL_NAMES}
    out["iv_skew_30d"] = _iv_skew_30d(g)
    out["iv_term_slope"] = _iv_term_slope(g)
    out["iv_convexity_30d"] = _iv_convexity_30d(g)
    out["cw_vol_spread"] = _cw_vol_spread(g)

    atm30 = _atm_iv_30(g)
    if hv is not None and np.isfinite(hv) and float(hv) > 0 and np.isfinite(atm30) and atm30 > 0:
        out["vmp"] = float(np.log(float(hv)) - np.log(atm30))

    m30 = _mf_moments_at_days(g, 30)
    m365 = _mf_moments_at_days(g, 365)
    out["mfiv_30"] = m30["mfiv"]
    out["mfis_30"] = m30["mfis"]
    out["mfik_30"] = m30["mfik"]
    out["mfiv_365"] = m365["mfiv"]
    out["mfis_365"] = m365["mfis"]
    out["mfik_365"] = m365["mfik"]
    if hv is not None and np.isfinite(hv) and np.isfinite(out["mfiv_30"]):
        out["vrp_30"] = float(out["mfiv_30"]) - float(hv)
    if np.isfinite(m365["mfis"]) and np.isfinite(m30["mfis"]):
        out["rns_term_spread"] = float(m365["mfis"] - m30["mfis"])

    out["svix2_30"] = _svix2_at_days(g, 30)
    out["surface_dispersion"] = _surface_dispersion(g)
    out["surface_quality"] = _surface_quality(g)

    if (
        option_volume is not None
        and equity_volume is not None
        and np.isfinite(option_volume)
        and np.isfinite(equity_volume)
        and float(equity_volume) > 0
    ):
        out["os_ratio"] = float(option_volume) / float(equity_volume)
    if borrow is not None and np.isfinite(borrow):
        out["borrow_rate"] = float(borrow)

    # ATM levels for An–Ang–Bali–Cakici ΔIV (filled later across month-ends).
    out["_atm_call_30"] = extract_grid_point(g, days=30, delta=50, cp_flag="C")
    out["_atm_put_30"] = extract_grid_point(g, days=30, delta=-50, cp_flag="P")
    return out


def _month_end_mask(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates)
    # Last observation date within each calendar month (trading-day month-end).
    ym = d.dt.to_period("M")
    last = d.groupby(ym).transform("max")
    return d == last


def _index_aux(
    table: pd.DataFrame | Mapping[str, Any] | None,
    value_col: str,
) -> dict[str, Any]:
    """Pre-index an aux table for exact and PIT as-of lookup.

    B1: the aux tables (``sp500_hv``, ``om_opvold``, ``om_borrate``) run to
    tens of millions of rows; a per-(secid, date)-group boolean-mask scan
    over the full table (the prior behavior of ``_lookup_aux``) is
    quadratic in the number of month-end groups and is infeasible at
    campaign scale. Build the lookup once per call to
    :func:`compute_surface_signals_panel` instead.

    Returns ``{"exact": {(secid, date): value}, "asof": {secid: [(date, value), ...]}}``
    with each asof list sorted by date. Exact hits prefer the map; misses
    fall back to the last observation at or before the query date (no bfill).

    If a table carries multiple rows per ``(secid, date)`` (e.g. HV or
    borrow rate reported across several tenors), the caller is expected to
    have already filtered to the desired tenor; this function otherwise
    keeps the last row encountered.
    """
    import bisect

    if table is None:
        return {"exact": {}, "asof": {}}
    df = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
    if df.empty:
        return {"exact": {}, "asof": {}}
    col = value_col
    if col not in df.columns:
        candidates = [c for c in df.columns if c not in ("secid", "date")]
        if len(candidates) == 1:
            col = candidates[0]
        else:
            return {"exact": {}, "asof": {}}
    dates = pd.to_datetime(df["date"])
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
    secids = df["secid"].to_numpy()
    exact: dict[tuple[Any, pd.Timestamp], float] = {}
    asof_raw: dict[Any, list[tuple[pd.Timestamp, float]]] = {}
    for sid, ts, v in zip(secids, dates, vals):
        if not np.isfinite(v):
            continue
        t = pd.Timestamp(ts)
        exact[(sid, t)] = float(v)
        asof_raw.setdefault(sid, []).append((t, float(v)))
    asof: dict[Any, list[tuple[pd.Timestamp, float]]] = {}
    for sid, pairs in asof_raw.items():
        pairs.sort(key=lambda x: x[0])
        # Dedup by date keeping last.
        dedup: list[tuple[pd.Timestamp, float]] = []
        for t, v in pairs:
            if dedup and dedup[-1][0] == t:
                dedup[-1] = (t, v)
            else:
                dedup.append((t, v))
        asof[sid] = dedup
    # bisect imported for _lookup_indexed closure clarity; kept module-level via caller
    _ = bisect
    return {"exact": exact, "asof": asof}


def _lookup_indexed(
    idx: Mapping[str, Any] | Mapping[tuple[Any, pd.Timestamp], float],
    secid: Any,
    date: pd.Timestamp,
) -> float | None:
    import bisect

    ts = pd.Timestamp(date)
    if not idx:
        return None
    # New shape: {"exact", "asof"}
    if "exact" in idx and "asof" in idx:
        exact = idx["exact"]
        hit = exact.get((secid, ts))
        if hit is not None:
            return float(hit)
        series = idx["asof"].get(secid) or []
        if not series:
            return None
        keys = [t for t, _ in series]
        pos = bisect.bisect_right(keys, ts) - 1
        if pos < 0:
            return None
        return float(series[pos][1])
    # Legacy flat map
    return idx.get((secid, ts))  # type: ignore[arg-type]


def _grouped_signal_rows(
    surface_df: pd.DataFrame,
    *,
    hv: pd.DataFrame | Mapping[str, Any] | None,
    option_volume: pd.DataFrame | Mapping[str, Any] | None,
    equity_volume: pd.DataFrame | Mapping[str, Any] | None,
    borrow: pd.DataFrame | Mapping[str, Any] | None,
    month_end_only: bool,
) -> pd.DataFrame:
    """Per-(secid, date) signal rows from a raw long surface, *before* the
    cross-sectional (``mw_xs``) and per-secid time-series (``d_iv_*_1m``)
    passes in :func:`_finalize_signals_panel`. Secid-independent, so callers
    that must bound peak memory (a full-universe raw surface load is the
    dominant cost, not this per-group step) can call this once per secid
    batch and concatenate the resulting small per-group rows before the
    single full-pool finalize pass.
    """
    if surface_df is None or len(surface_df) == 0:
        return pd.DataFrame()

    df = surface_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["secid"] = df["secid"]
    if month_end_only:
        keep_dates = df.loc[_month_end_mask(df["date"]), "date"].unique()
        df = df[df["date"].isin(keep_dates)]

    hv_idx = _index_aux(hv, "hv")
    opvol_idx = _index_aux(option_volume, "option_volume")
    eqvol_idx = _index_aux(equity_volume, "equity_volume")
    borrow_idx = _index_aux(borrow, "borrow_rate")

    rows: list[dict[str, Any]] = []
    for (secid, date), g in df.groupby(["secid", "date"], sort=True):
        ts = pd.Timestamp(date)
        sig = _signals_for_group(
            g,
            hv=_lookup_indexed(hv_idx, secid, ts),
            option_volume=_lookup_indexed(opvol_idx, secid, ts),
            equity_volume=_lookup_indexed(eqvol_idx, secid, ts),
            borrow=_lookup_indexed(borrow_idx, secid, ts),
        )
        sig["secid"] = secid
        sig["date"] = ts
        rows.append(sig)

    return pd.DataFrame(rows)


