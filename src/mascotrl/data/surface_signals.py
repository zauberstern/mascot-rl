"""Literature-backed IV surface signals for equity allocation (Phase E).

Pure constructors operate on a standardized OptionMetrics-style surface table
(columns: secid, date, days, delta, cp_flag, impl_volatility, impl_strike,
impl_premium, dispersion) without requiring the lake. Lake materialization is
a thin DuckDB pass over ``vol_surface`` partitions.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from mascotrl.data.implied_moments import compute_mf_moments

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

# OptionMetrics standardized delta-tenor grid (Kelly 2026 / OM volsurfd).
KELLY_TENORS: tuple[int, ...] = (10, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730)
KELLY_DELTAS_PUT: tuple[int, ...] = tuple(range(-90, -5, 5))  # 17
KELLY_DELTAS_CALL: tuple[int, ...] = tuple(range(10, 95, 5))  # 17
GRID_POINTS_PER_DAY: int = len(KELLY_TENORS) * (
    len(KELLY_DELTAS_PUT) + len(KELLY_DELTAS_CALL)
)  # 374


def validate_kelly_grid_schema(
    *,
    tenors: Sequence[int] = KELLY_TENORS,
    deltas_put: Sequence[int] = KELLY_DELTAS_PUT,
    deltas_call: Sequence[int] = KELLY_DELTAS_CALL,
    cube_shape: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Assert Kelly axes match OM delta-tenor nodes; optional cube shape check.

    Returns a small metadata dict for logging / tests. Raises ``ValueError``
    on schema mismatch.
    """
    tenors_t = tuple(int(t) for t in tenors)
    d_put = tuple(int(d) for d in deltas_put)
    d_call = tuple(int(d) for d in deltas_call)
    if tenors_t != KELLY_TENORS:
        raise ValueError(f"Kelly tenors must equal OM grid, got {tenors_t}")
    if d_put != KELLY_DELTAS_PUT:
        raise ValueError(f"Kelly put deltas must equal OM grid, got {d_put}")
    if d_call != KELLY_DELTAS_CALL:
        raise ValueError(f"Kelly call deltas must equal OM grid, got {d_call}")
    n_del = len(d_put) + len(d_call)
    expected = (len(tenors_t), n_del)
    if cube_shape is not None:
        if len(cube_shape) != 4:
            raise ValueError(f"Kelly cube must be (T,K,11,34), got shape={cube_shape}")
        if cube_shape[2:] != expected:
            raise ValueError(
                f"Kelly cube tenor/delta axes {cube_shape[2:]} != {expected}"
            )
    return {
        "n_tenors": len(tenors_t),
        "n_deltas": n_del,
        "grid_points_per_day": len(tenors_t) * n_del,
        "ffill_axis": "date",
        "ffill_causal": True,
    }


CW_DAYS: tuple[int, ...] = (30, 60, 91)
CW_DELTAS: tuple[int, ...] = (20, 25, 30, 40, 50)

SURFACE_SIGNAL_NAMES: tuple[str, ...] = (
    "iv_skew_30d",
    "iv_term_slope",
    "iv_convexity_30d",
    "cw_vol_spread",
    "vmp",
    "mfiv_30",
    "mfis_30",
    "mfik_30",
    "mfiv_365",
    "mfis_365",
    "mfik_365",
    "rns_term_spread",
    "svix2_30",
    "mw_xs",
    "d_iv_call_1m",
    "d_iv_put_1m",
    "surface_dispersion",
    "surface_quality",
    "os_ratio",
    "borrow_rate",
    "d_iv_term_slope_5d",
    "d_iv_skew_5d",
    "vrp_30",
)


_NAN = float("nan")


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


def _finalize_signals_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional (``mw_xs``) and per-secid time-series (``d_iv_*_1m``)
    passes over the *full-pool* per-(secid, date) rows from
    :func:`_grouped_signal_rows`. Must run once over every secid in the
    pool (``mw_xs`` is a per-date cross-sectional demeaning), never per
    secid-batch.
    """
    if panel.empty:
        return panel

    # Cross-sectional Martin–Wagner: mw_xs = 0.5 (svix2_i − xs mean) after per-name svix2.
    panel["mw_xs"] = _NAN
    for _dt, idx in panel.groupby("date").groups.items():
        ix = list(idx)
        sv = panel.loc[ix, "svix2_30"].to_numpy(dtype=float)
        finite = np.isfinite(sv)
        if finite.any():
            mu = float(np.nanmean(sv[finite]))
            panel.loc[ix, "mw_xs"] = 0.5 * (panel.loc[ix, "svix2_30"].to_numpy(dtype=float) - mu)

    # An–Ang–Bali–Cakici month-over-month ΔIV: prior month-end only (PIT).
    panel = panel.sort_values(["secid", "date"]).reset_index(drop=True)
    panel["d_iv_call_1m"] = _NAN
    panel["d_iv_put_1m"] = _NAN
    for _sid, idx in panel.groupby("secid").groups.items():
        ix = list(idx)
        sub = panel.loc[ix]
        call = sub["_atm_call_30"].to_numpy(dtype=float)
        put = sub["_atm_put_30"].to_numpy(dtype=float)
        d_call = np.full(len(ix), _NAN)
        d_put = np.full(len(ix), _NAN)
        if len(ix) >= 2:
            d_call[1:] = call[1:] - call[:-1]
            d_put[1:] = put[1:] - put[:-1]
            # Invalidate if either leg missing.
            bad_c = ~(np.isfinite(call[1:]) & np.isfinite(call[:-1]))
            bad_p = ~(np.isfinite(put[1:]) & np.isfinite(put[:-1]))
            d_call[1:][bad_c] = _NAN
            d_put[1:][bad_p] = _NAN
        panel.loc[ix, "d_iv_call_1m"] = d_call
        panel.loc[ix, "d_iv_put_1m"] = d_put

    # 5-observation first differences of geometry (daily panels); NaN if lag missing.
    # Strict PIT: observation lag only (no bfill / zero-fill).
    panel["d_iv_term_slope_5d"] = _NAN
    panel["d_iv_skew_5d"] = _NAN
    for _sid, idx in panel.groupby("secid").groups.items():
        ix = list(idx)
        if len(ix) < 6:
            continue
        term = panel.loc[ix, "iv_term_slope"].to_numpy(dtype=float)
        skew = panel.loc[ix, "iv_skew_30d"].to_numpy(dtype=float)
        d_term = np.full(len(ix), _NAN)
        d_skew = np.full(len(ix), _NAN)
        d_term[5:] = term[5:] - term[:-5]
        d_skew[5:] = skew[5:] - skew[:-5]
        bad_t = ~(np.isfinite(term[5:]) & np.isfinite(term[:-5]))
        bad_s = ~(np.isfinite(skew[5:]) & np.isfinite(skew[:-5]))
        d_term[5:][bad_t] = _NAN
        d_skew[5:][bad_s] = _NAN
        panel.loc[ix, "d_iv_term_slope_5d"] = d_term
        panel.loc[ix, "d_iv_skew_5d"] = d_skew

    drop_aux = [c for c in ("_atm_call_30", "_atm_put_30") if c in panel.columns]
    panel = panel.drop(columns=drop_aux)
    ordered = ["secid", "date", *SURFACE_SIGNAL_NAMES]
    # Preserve any extra columns only if already in SURFACE_SIGNAL_NAMES.
    return panel.loc[:, [c for c in ordered if c in panel.columns]].reset_index(drop=True)


def compute_surface_signals_panel(
    surface_df: pd.DataFrame,
    *,
    hv: pd.DataFrame | Mapping[str, Any] | None = None,
    option_volume: pd.DataFrame | Mapping[str, Any] | None = None,
    equity_volume: pd.DataFrame | Mapping[str, Any] | None = None,
    borrow: pd.DataFrame | Mapping[str, Any] | None = None,
    month_end_only: bool = True,
) -> pd.DataFrame:
    """Compute per-(secid, date) surface signals. PIT-safe for ``d_iv_*``.

    Parameters
    ----------
    surface_df :
        Long surface with columns ``secid, date, days, delta, cp_flag,
        impl_volatility`` and optionally ``impl_strike, impl_premium, dispersion``.
    hv / option_volume / equity_volume / borrow :
        Optional long tables keyed by ``(secid, date)``.
    month_end_only :
        If True, keep the last trading day of each calendar month.
    """
    if surface_df is None or len(surface_df) == 0:
        cols = ["secid", "date", *SURFACE_SIGNAL_NAMES]
        return pd.DataFrame(columns=cols)
    rows_panel = _grouped_signal_rows(
        surface_df,
        hv=hv,
        option_volume=option_volume,
        equity_volume=equity_volume,
        borrow=borrow,
        month_end_only=month_end_only,
    )
    if rows_panel.empty:
        return rows_panel
    return _finalize_signals_panel(rows_panel)


def build_kelly_iv_images(
    surface_df: pd.DataFrame,
    *,
    secids: Sequence[Any],
    dates: Sequence[Any],
    tenors: Sequence[int] = KELLY_TENORS,
    deltas_put: Sequence[int] = KELLY_DELTAS_PUT,
    deltas_call: Sequence[int] = KELLY_DELTAS_CALL,
) -> np.ndarray:
    """Build Kelly-style IV image tensor ``(n_dates, n_secids, n_tenors, n_deltas)``.

    Delta axis is puts then calls (17 + 17 = 34). Missing nodes are NaN.
    """
    validate_kelly_grid_schema(
        tenors=tuple(int(t) for t in tenors),
        deltas_put=tuple(int(d) for d in deltas_put),
        deltas_call=tuple(int(d) for d in deltas_call),
    )
    tenors_t = tuple(int(t) for t in tenors)
    d_put = tuple(int(d) for d in deltas_put)
    d_call = tuple(int(d) for d in deltas_call)
    delta_axis = d_put + d_call
    n_dates = len(dates)
    n_sec = len(secids)
    n_ten = len(tenors_t)
    n_del = len(delta_axis)
    out = np.full((n_dates, n_sec, n_ten, n_del), np.nan, dtype=np.float64)
    if surface_df is None or len(surface_df) == 0 or n_dates == 0 or n_sec == 0:
        return out

    df = surface_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    date_index = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    sec_index = {s: i for i, s in enumerate(secids)}
    tenor_index = {t: i for i, t in enumerate(tenors_t)}
    # Map (cp, delta) → axis index
    delta_index: dict[tuple[str, int], int] = {}
    for i, d in enumerate(d_put):
        delta_index[("P", int(d))] = i
    for i, d in enumerate(d_call):
        delta_index[("C", int(d))] = len(d_put) + i

    df = df[df["secid"].isin(sec_index) & df["date"].isin(date_index)]
    if df.empty:
        return out

    # Vectorized index resolution: a Python-level `for i in range(len(df))`
    # loop over per-row `.iloc` lookups is O(rows) with heavy pandas
    # overhead per access -- at production scale (K names x years of daily
    # 374-node grids) that is millions of rows and takes tens of minutes.
    # Map every column to its axis index (or NaN) in one vectorized pass,
    # then assign via numpy fancy indexing (last-write-wins on duplicate
    # indices, matching the prior loop's sequential-assignment semantics).
    days = pd.to_numeric(df["days"], errors="coerce")
    deltas = pd.to_numeric(df["delta"], errors="coerce")
    ivs = pd.to_numeric(df["impl_volatility"], errors="coerce")
    cps = df["cp_flag"].map(_as_cp)

    valid_iv = ivs.notna() & (ivs > 0) & days.notna() & deltas.notna()
    if not bool(valid_iv.any()):
        return out
    days_i = days[valid_iv].astype("int64")
    deltas_i = deltas[valid_iv].astype("int64")
    cp_v = cps[valid_iv]
    delta_key = list(zip(cp_v.tolist(), deltas_i.tolist()))

    ti = days_i.map(tenor_index)
    di = pd.Series(delta_key, index=days_i.index).map(delta_index)
    si = df.loc[valid_iv, "secid"].map(sec_index)
    dai = df.loc[valid_iv, "date"].map(date_index)

    node_ok = ti.notna() & di.notna() & si.notna() & dai.notna()
    if not bool(node_ok.any()):
        return out
    out[
        dai[node_ok].to_numpy(dtype=np.int64),
        si[node_ok].to_numpy(dtype=np.int64),
        ti[node_ok].to_numpy(dtype=np.int64),
        di[node_ok].to_numpy(dtype=np.int64),
    ] = ivs[valid_iv][node_ok].to_numpy(dtype=np.float64)
    return out


def cache_surface_signals(panel: pd.DataFrame, path: str | Path) -> None:
    """Write signal panel to parquet."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = panel.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
    out.to_parquet(p, index=False)


def load_surface_signals(path: str | Path) -> pd.DataFrame:
    """Load signal panel from parquet."""
    df = pd.read_parquet(Path(path))
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def surface_signals_cache_fingerprint(
    *,
    secids: Sequence[Any],
    start: str,
    end: str,
    signal_names: Sequence[str] | None = None,
) -> str:
    """Sha256 key for shared surface-signal parquet reuse across campaign arms."""
    import hashlib

    names = sorted(str(n) for n in (signal_names if signal_names is not None else SURFACE_SIGNAL_NAMES))
    sec_keys = sorted(_canonical_secid_key(s) for s in secids)
    payload = "\n".join(
        [
            ",".join(sec_keys),
            str(start),
            str(end),
            ",".join(names),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _shared_surface_cache_paths(cache_dir: str | Path, fingerprint: str) -> tuple[Path, Path]:
    root = Path(cache_dir)
    return root / f"{fingerprint}.parquet", root / f"{fingerprint}.meta.json"


def _load_shared_surface_cache(
    cache_dir: str | Path,
    fingerprint: str,
) -> pd.DataFrame | None:
    """Load fingerprint-keyed panel or return None on miss.

    Fail-closed when a cache artifact exists but is corrupt / mismatched:
    never silently rebuild over a broken shared file.
    """
    import json

    parquet_path, meta_path = _shared_surface_cache_paths(cache_dir, fingerprint)
    if not parquet_path.is_file():
        return None
    if not meta_path.is_file():
        raise RuntimeError(
            f"surface cache corrupt: missing meta for {parquet_path}"
        )
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"surface cache corrupt: unreadable meta {meta_path}: {exc}"
        ) from exc
    if str(meta.get("fingerprint") or "") != str(fingerprint):
        raise RuntimeError(
            f"surface cache corrupt: fingerprint mismatch at {meta_path}"
        )
    try:
        df = load_surface_signals(parquet_path)
    except Exception as exc:
        raise RuntimeError(
            f"surface cache corrupt: unreadable parquet {parquet_path}: {exc}"
        ) from exc
    required = {"secid", "date"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"surface cache corrupt: missing columns {sorted(missing)} in {parquet_path}"
        )
    _LOG.info("surface_signals cache hit path=%s rows=%d", parquet_path, len(df))
    return df


def _write_shared_surface_cache(
    cache_dir: str | Path,
    fingerprint: str,
    panel: pd.DataFrame,
) -> None:
    import json

    parquet_path, meta_path = _shared_surface_cache_paths(cache_dir, fingerprint)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    cache_surface_signals(panel, parquet_path)
    meta_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "n_rows": int(len(panel)),
                "columns": list(panel.columns),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _LOG.info("surface_signals cache wrote path=%s rows=%d", parquet_path, len(panel))


def _canonical_secid_key(secid: Any) -> str:
    """Stable string key so ``100892`` and ``100892.0`` align to the same column.

    DuckDB / parquet often yield float secids; equity panels use ints. A naive
    ``str(secid)`` then makes ``align_signals_to_panel`` reindex to all-NaN
    columns and dyn_dii_opt eligibility collapses to zero names.
    """
    if secid is None:
        return "None"
    try:
        if isinstance(secid, (float, np.floating)) and not np.isfinite(secid):
            return str(secid)
        return str(int(secid))
    except (TypeError, ValueError):
        return str(secid)


def align_signals_to_panel(
    signals: pd.DataFrame,
    dates: Sequence[Any],
    secids: Sequence[Any],
    *,
    lag_days: int = 1,
    signal_names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Forward-fill month-end signals onto a daily ``(T, K)`` panel, PIT-safe.

    ``signals`` is the long month-end frame from
    :func:`materialize_surface_signals_from_lake` (columns ``secid, date,
    <signal columns>``). Each month-end value becomes visible only
    ``lag_days`` calendar days after the month-end date it describes (a
    publication lag), and then holds (forward-fills) until the next
    release. A daily date ``d`` therefore only ever sees a signal computed
    from a month-end ``<= d - lag_days``, matching the OptionMetrics
    end-of-day availability convention.

    Returns ``{signal_name: (T, K) float64 array}`` aligned to ``dates`` x
    ``secids``, with ``NaN`` where no published value is available yet.
    """
    dates_idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    t = len(dates_idx)
    k = len(secids)
    names = list(signal_names) if signal_names is not None else list(SURFACE_SIGNAL_NAMES)
    if signals is None or len(signals) == 0:
        return {name: np.full((t, k), np.nan, dtype=np.float64) for name in names}

    sig = signals.copy()
    sig["date"] = pd.to_datetime(sig["date"])
    # Normalize secid so float/int parquet variants share one key space.
    sig["secid"] = sig["secid"].map(_canonical_secid_key)
    # Publication lag: a month-end value is not knowable until lag_days later.
    sig["_avail"] = sig["date"] + pd.Timedelta(days=int(lag_days))

    sec_str = [_canonical_secid_key(s) for s in secids]
    out: dict[str, np.ndarray] = {}
    for name in names:
        if name not in sig.columns:
            out[name] = np.full((t, k), np.nan, dtype=np.float64)
            continue
        wide = sig.pivot_table(index="_avail", columns="secid", values=name, aggfunc="last")
        wide.columns = [_canonical_secid_key(c) for c in wide.columns]
        wide = wide.reindex(columns=sec_str)
        # Union index (avail dates + target dates) then ffill so a value
        # published between two requested dates is still visible on the
        # next requested date, before restricting back to `dates`.
        union_idx = wide.index.union(dates_idx).sort_values()
        wide = wide.reindex(union_idx).ffill()
        wide = wide.reindex(dates_idx)
        out[name] = wide.to_numpy(dtype=np.float64)
    return out


def align_signals_to_slots(
    signals: pd.DataFrame,
    dates: Sequence[Any],
    slots_rows: Sequence[Sequence[Any]],
    *,
    lag_days: int = 1,
    signal_names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """PIT-align surface signals onto a *dynamic* slot occupancy schedule.

    Unlike :func:`align_signals_to_panel` (fixed secid identity per column),
    this gathers the published value of whichever secid occupies slot ``k``
    on date ``t``. Inactive slots (``None``) are NaN. Publication lag and
    month-end forward-fill semantics match the static panel helper.
    """
    dates_idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    t_n = len(dates_idx)
    if len(slots_rows) != t_n:
        raise ValueError(
            f"slots_rows length {len(slots_rows)} != len(dates)={t_n}"
        )
    k = len(slots_rows[0]) if slots_rows else 0
    for i, row in enumerate(slots_rows):
        if len(row) != k:
            raise ValueError(f"slots_rows[{i}] length {len(row)} != k={k}")

    names = list(signal_names) if signal_names is not None else list(SURFACE_SIGNAL_NAMES)
    empty = {name: np.full((t_n, k), np.nan, dtype=np.float64) for name in names}
    if signals is None or len(signals) == 0 or k == 0:
        return empty

    # Universe of every secid that ever occupies a slot.
    all_secids: list[Any] = []
    seen: set[str] = set()
    for row in slots_rows:
        for sid in row:
            if sid is None:
                continue
            key = str(sid)
            if key not in seen:
                seen.add(key)
                all_secids.append(sid)
    if not all_secids:
        return empty

    panel = align_signals_to_panel(
        signals,
        dates_idx,
        all_secids,
        lag_days=lag_days,
        signal_names=names,
    )
    col_of = {_canonical_secid_key(s): i for i, s in enumerate(all_secids)}
    out: dict[str, np.ndarray] = {}
    for name in names:
        wide = panel[name]
        slotted = np.full((t_n, k), np.nan, dtype=np.float64)
        for t, row in enumerate(slots_rows):
            for j, sid in enumerate(row):
                if sid is None:
                    continue
                col = col_of.get(_canonical_secid_key(sid))
                if col is None:
                    continue
                slotted[t, j] = wide[t, col]
        out[name] = slotted
    return out


def _load_vol_surface_raw(
    lake_root: str | Path,
    *,
    secids: Sequence[Any],
    start: str,
    end: str,
) -> pd.DataFrame:
    """DuckDB scan of ``vol_surface`` filtered to ``secids`` / ``[start, end]``.

    Shared by :func:`materialize_surface_signals_from_lake` (month-end
    signal computation) and :func:`materialize_kelly_iv_images_from_lake`
    (raw per-date IV grid), so both read the same filtered rows once.
    Raises ``FileNotFoundError`` when the lake root or ``vol_surface`` tree
    is absent. Prefers a filtered parquet scan (secids + date range) over
    loading all rows.
    """
    root = Path(lake_root)
    vs = root / "vol_surface"
    if not root.exists() or not vs.exists():
        raise FileNotFoundError(f"vol_surface lake not found under {root}")

    glob = (vs / "year=*" / "month=*" / "data_0.parquet").as_posix()
    # Also accept any *.parquet under hive partitions.
    alt_glob = (vs / "*" / "*" / "*.parquet").as_posix()
    id_list = ", ".join(str(int(s)) for s in secids)
    if not id_list:
        raise ValueError("secids must be non-empty")

    import duckdb

    # B1 perf: the `date` predicate alone does not let DuckDB skip
    # `year=*/month=*` partition files it still has to open every file in
    # the glob to evaluate it. Filtering directly on the hive partition
    # columns (exposed by `hive_partitioning=1`) lets it prune files
    # outside [start, end] at the year granularity before reading them,
    # which matters at ~17GB / 264 monthly partitions.
    year_lo = int(str(start)[:4])
    year_hi = int(str(end)[:4])

    # An unconfigured connection defaults to DuckDB's own memory/thread
    # auto-detection (a large fraction of total system RAM and all cores),
    # which at a full-universe secid pool (hundreds of names x ~a decade)
    # was observed to balloon host memory and trigger an OOM kill of the
    # whole session. Honor the same env-configured ceiling the lake builder
    # uses so this scan degrades to disk spill instead of OOM.
    mem_limit = os.environ.get("MASCOTRL_DUCKDB_MAX_MEMORY", "4GB")
    n_threads = int(os.environ.get("MASCOTRL_DUCKDB_THREADS", "4"))

    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit = '{mem_limit}';")
        con.execute(f"SET threads TO {n_threads};")
        sql = f"""
        SELECT
            TRY_CAST(secid AS BIGINT) AS secid,
            CAST(date AS DATE) AS date,
            TRY_CAST(days AS BIGINT) AS days,
            TRY_CAST(delta AS BIGINT) AS delta,
            CAST(cp_flag AS VARCHAR) AS cp_flag,
            TRY_CAST(impl_volatility AS DOUBLE) AS impl_volatility,
            TRY_CAST(impl_strike AS DOUBLE) AS impl_strike,
            TRY_CAST(impl_premium AS DOUBLE) AS impl_premium,
            TRY_CAST(dispersion AS DOUBLE) AS dispersion
        FROM read_parquet('{alt_glob}', hive_partitioning=1, union_by_name=true)
        WHERE TRY_CAST(secid AS BIGINT) IN ({id_list})
          AND CAST(date AS DATE) BETWEEN DATE '{start}' AND DATE '{end}'
          AND TRY_CAST(year AS BIGINT) BETWEEN {year_lo} AND {year_hi}
        """
        try:
            surface = con.execute(sql).fetch_df()
        except Exception as e:
            # A10: record why the hive glob failed before falling back to
            # explicit data_0 naming; if the fallback also fails the
            # exception propagates (fail-closed).
            _LOG.warning("vol_surface hive glob read failed (%s); retrying with %s", e, glob)
            sql2 = sql.replace(alt_glob, glob)
            surface = con.execute(sql2).fetch_df()
    finally:
        con.close()
    return surface


DEFAULT_SECID_BATCH_SIZE = 60



def _load_surface_aux_from_lake(
    lake_root: str | Path,
    *,
    secids: Sequence[Any],
    start: str,
    end: str,
) -> dict[str, pd.DataFrame | None]:
    """Load hv_21 / om_opvold / om_borrate / equity volume for unscored signals.

    ``hv`` prefers annualized trailing stdev (21) from equity returns when the
    security-return panel is available; falls back to lake ``sp500_hv`` tenor 30
    (nearest OM historical-vol tenor to the ATM-30 surface).
    """
    root = Path(lake_root)
    secid_set = set(int(s) for s in secids)
    out: dict[str, pd.DataFrame | None] = {
        "hv": None,
        "option_volume": None,
        "equity_volume": None,
        "borrow": None,
    }

    # --- HV (annualized stdev convention) ---
    hv_df = None
    try:
        from mascotrl.data.equity_panel import load_sp500_security_returns

        raw = load_sp500_security_returns(root, start=start, end=end)
        if raw is not None and len(raw) and "secid" in raw.columns:
            raw = raw.copy()
            raw["secid"] = pd.to_numeric(raw["secid"], errors="coerce")
            raw = raw[raw["secid"].isin(secid_set)]
            raw["date"] = pd.to_datetime(raw["date"])
            ret_col = "stk_ret" if "stk_ret" in raw.columns else "return"
            parts = []
            for sid, g in raw.groupby("secid", sort=False):
                g = g.sort_values("date")
                r = pd.to_numeric(g[ret_col], errors="coerce")
                hv = r.rolling(21, min_periods=21).std(ddof=1) * float(np.sqrt(252.0))
                parts.append(pd.DataFrame({"secid": sid, "date": g["date"].to_numpy(), "hv": hv.to_numpy()}))
            if parts:
                hv_df = pd.concat(parts, ignore_index=True)
                hv_df = hv_df[np.isfinite(hv_df["hv"])]
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("surface aux hv_21 from returns failed: %s", exc)

    if hv_df is None or hv_df.empty:
        hv_path = root / "macro" / "sp500_hv.parquet"
        if hv_path.is_file():
            try:
                hv = pd.read_parquet(hv_path, columns=["secid", "date", "days", "volatility"])
                hv["secid"] = pd.to_numeric(hv["secid"], errors="coerce")
                hv["date"] = pd.to_datetime(hv["date"])
                hv["days"] = pd.to_numeric(hv["days"], errors="coerce")
                hv = hv[
                    hv["secid"].isin(secid_set)
                    & (hv["date"] >= pd.Timestamp(start))
                    & (hv["date"] <= pd.Timestamp(end))
                    & (hv["days"] == 30)
                ]
                hv_df = hv.rename(columns={"volatility": "hv"})[["secid", "date", "hv"]]
                hv_df["hv"] = pd.to_numeric(hv_df["hv"], errors="coerce")
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("surface aux sp500_hv load failed: %s", exc)
    out["hv"] = hv_df if hv_df is not None and len(hv_df) else None

    # --- option volume ---
    op_path = root / "macro" / "om_opvold.parquet"
    if op_path.is_file():
        try:
            op = pd.read_parquet(op_path, columns=["secid", "date", "volume"])
            op["secid"] = pd.to_numeric(op["secid"], errors="coerce")
            op["date"] = pd.to_datetime(op["date"])
            op = op[
                op["secid"].isin(secid_set)
                & (op["date"] >= pd.Timestamp(start))
                & (op["date"] <= pd.Timestamp(end))
            ]
            agg = op.groupby(["secid", "date"], as_index=False)["volume"].sum()
            out["option_volume"] = agg.rename(columns={"volume": "option_volume"})
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("surface aux om_opvold load failed: %s", exc)

    # --- borrow ---
    br_path = root / "macro" / "om_borrate.parquet"
    if br_path.is_file():
        try:
            br = pd.read_parquet(br_path, columns=["secid", "date", "days", "borrowrate"])
            br["secid"] = pd.to_numeric(br["secid"], errors="coerce")
            br["date"] = pd.to_datetime(br["date"])
            br["days"] = pd.to_numeric(br["days"], errors="coerce")
            br = br[
                br["secid"].isin(secid_set)
                & (br["date"] >= pd.Timestamp(start))
                & (br["date"] <= pd.Timestamp(end))
                & (br["days"] == 30)
            ]
            # OM uses -99.99 as missing sentinel.
            br["borrowrate"] = pd.to_numeric(br["borrowrate"], errors="coerce")
            br.loc[br["borrowrate"] <= -90.0, "borrowrate"] = np.nan
            out["borrow"] = br.rename(columns={"borrowrate": "borrow_rate"})[
                ["secid", "date", "borrow_rate"]
            ]
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("surface aux om_borrate load failed: %s", exc)

    # --- equity volume (for os_ratio) ---
    try:
        from mascotrl.data.equity_panel import load_sp500_security_returns

        raw = load_sp500_security_returns(root, start=start, end=end)
        if raw is not None and len(raw) and "volume" in raw.columns:
            raw = raw.copy()
            raw["secid"] = pd.to_numeric(raw["secid"], errors="coerce")
            raw = raw[raw["secid"].isin(secid_set)]
            raw["date"] = pd.to_datetime(raw["date"])
            eq = raw[["secid", "date", "volume"]].rename(columns={"volume": "equity_volume"})
            eq["equity_volume"] = pd.to_numeric(eq["equity_volume"], errors="coerce")
            out["equity_volume"] = eq
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("surface aux equity volume load failed: %s", exc)

    return out


def materialize_surface_signals_from_lake(
    lake_root: str | Path,
    *,
    secids: Sequence[Any],
    start: str,
    end: str,
    cache_path: str | Path | None = None,
    hv: pd.DataFrame | None = None,
    option_volume: pd.DataFrame | None = None,
    equity_volume: pd.DataFrame | None = None,
    borrow: pd.DataFrame | None = None,
    month_end_only: bool = True,
    secid_batch_size: int | None = DEFAULT_SECID_BATCH_SIZE,
) -> pd.DataFrame:
    """Load ``vol_surface`` via DuckDB and compute the month-end signal panel.

    ``secid_batch_size`` bounds peak memory: the raw per-quote surface scan
    (``_load_vol_surface_raw``) dominates memory use at a full-universe pool
    (hundreds of names x a decade of daily option chains materialized into
    one pandas DataFrame overflowed host RAM in production). When the pool
    exceeds this size the raw load + per-group signal step run in secid
    batches and are concatenated *before* the single full-pool
    cross-sectional finalize pass, so results are identical to the
    unbatched path (``mw_xs`` still cross-sections over every requested
    secid, never just one batch). Pass ``None`` or a value >= pool size to
    disable batching.

    When ``MASCOTRL_SURFACE_CACHE_DIR`` is set, a fingerprint-keyed parquet
    (sorted secids + date range + signal names) is reused across callers.
    Corrupt shared-cache artifacts raise rather than silently rebuilding.
    """
    secids_list = list(secids)
    if hv is None or option_volume is None or equity_volume is None or borrow is None:
        aux = _load_surface_aux_from_lake(
            lake_root, secids=secids_list, start=start, end=end
        )
        if hv is None:
            hv = aux.get("hv")
        if option_volume is None:
            option_volume = aux.get("option_volume")
        if equity_volume is None:
            equity_volume = aux.get("equity_volume")
        if borrow is None:
            borrow = aux.get("borrow")
    signal_names = list(SURFACE_SIGNAL_NAMES)
    shared_dir = str(os.environ.get("MASCOTRL_SURFACE_CACHE_DIR") or "").strip()
    shared_fp: str | None = None
    if shared_dir:
        shared_fp = surface_signals_cache_fingerprint(
            secids=secids_list,
            start=start,
            end=end,
            signal_names=signal_names,
        )
        hit = _load_shared_surface_cache(shared_dir, shared_fp)
        if hit is not None:
            if cache_path is not None:
                cache_surface_signals(hit, cache_path)
            return hit

    if not secid_batch_size or len(secids_list) <= secid_batch_size:
        surface = _load_vol_surface_raw(lake_root, secids=secids_list, start=start, end=end)
        panel = compute_surface_signals_panel(
            surface,
            hv=hv,
            option_volume=option_volume,
            equity_volume=equity_volume,
            borrow=borrow,
            month_end_only=month_end_only,
        )
    else:
        parts: list[pd.DataFrame] = []
        for i in range(0, len(secids_list), secid_batch_size):
            batch = secids_list[i : i + secid_batch_size]
            surface = _load_vol_surface_raw(lake_root, secids=batch, start=start, end=end)
            rows = _grouped_signal_rows(
                surface,
                hv=hv,
                option_volume=option_volume,
                equity_volume=equity_volume,
                borrow=borrow,
                month_end_only=month_end_only,
            )
            del surface
            if not rows.empty:
                parts.append(rows)
        rows_panel = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if rows_panel.empty:
            cols = ["secid", "date", *SURFACE_SIGNAL_NAMES]
            panel = pd.DataFrame(columns=cols)
        else:
            panel = _finalize_signals_panel(rows_panel)
    if shared_dir and shared_fp is not None:
        _write_shared_surface_cache(shared_dir, shared_fp, panel)
    if cache_path is not None:
        cache_surface_signals(panel, cache_path)
    return panel


def _kelly_cache_fingerprint(
    *,
    secids: Sequence[Any],
    dates: Sequence[Any],
    start: str,
    end: str,
    forward_fill: bool,
) -> str:
    """Stable key so a resumed campaign can reuse a Kelly cube on disk."""
    import hashlib

    payload = {
        "secids": [_canonical_secid_key(s) for s in secids],
        "dates": [str(pd.Timestamp(d).date()) for d in dates],
        "start": str(start),
        "end": str(end),
        "forward_fill": bool(forward_fill),
        "tenors": list(KELLY_TENORS),
        "deltas_put": list(KELLY_DELTAS_PUT),
        "deltas_call": list(KELLY_DELTAS_CALL),
    }
    blob = repr(payload).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


def materialize_kelly_iv_images_from_lake(
    lake_root: str | Path,
    *,
    secids: Sequence[Any],
    dates: Sequence[Any],
    start: str,
    end: str,
    forward_fill: bool = True,
    secid_batch_size: int | None = DEFAULT_SECID_BATCH_SIZE,
    cache_path: str | Path | None = None,
) -> np.ndarray:
    """B3: real per-date Kelly IV-surface image tensor from the lake.

    Returns ``(T, K, 11, 34)`` (``T=len(dates)``, ``K=len(secids)``), the
    layout ``src.eval.ceiling_arms._select_kelly_image_batch`` expects for
    ``kelly_cnn``. ``forward_fill=True`` (default) causally fills a grid
    cell with the most recent *earlier* observed value when the exact
    date has no quote for that (tenor, delta, cp) node -- never from a
    later date, so no look-ahead is introduced.

    ``secid_batch_size`` bounds peak memory the same way
    :func:`materialize_surface_signals_from_lake` does: a single DuckDB
    ``fetch_df`` of every daily option-chain quote for K=100 over a decade
    OOMed the full campaign after the signal gate had already succeeded.
    Batches fill disjoint slices of the output cube and are identical to
    the unbatched path (per-secid grids do not cross-talk).

    ``cache_path`` (optional ``.npz``) stores the cube keyed by a fingerprint
    of secids/dates/window so a killed-and-resumed campaign does not re-pay
    multi-hour lake scans. Cache hit requires an adjacent ``.meta.json`` with
    a matching fingerprint.
    """
    secids_list = list(secids)
    dates_list = list(dates)
    n_t = len(dates_list)
    n_k = len(secids_list)
    n_ten = len(KELLY_TENORS)
    n_del = len(KELLY_DELTAS_PUT) + len(KELLY_DELTAS_CALL)
    cube = np.full((n_t, n_k, n_ten, n_del), np.nan, dtype=np.float64)
    if n_t == 0 or n_k == 0:
        return cube

    fp = _kelly_cache_fingerprint(
        secids=secids_list,
        dates=dates_list,
        start=start,
        end=end,
        forward_fill=forward_fill,
    )
    if cache_path is not None:
        cpath = Path(cache_path)
        meta_path = Path(str(cpath) + ".meta.json")
        if cpath.is_file() and meta_path.is_file():
            try:
                import json

                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("fingerprint") == fp:
                    loaded = np.load(cpath)
                    arr = loaded["cube"] if isinstance(loaded, np.lib.npyio.NpzFile) else loaded
                    if tuple(arr.shape) == (n_t, n_k, n_ten, n_del):
                        _LOG.info(
                            "kelly_images cache hit path=%s shape=%s",
                            cpath,
                            arr.shape,
                        )
                        return np.asarray(arr, dtype=np.float64)
            except Exception as exc:  # pragma: no cover - best-effort cache
                _LOG.warning("kelly_images cache unreadable (%s); rebuilding", exc)

    batch_size = secid_batch_size if secid_batch_size else n_k
    n_batches = (n_k + batch_size - 1) // batch_size
    for bi, i in enumerate(range(0, n_k, batch_size)):
        batch = secids_list[i : i + batch_size]
        _LOG.info(
            "kelly_images batch %d/%d secids=%d..%d T=%d",
            bi + 1,
            n_batches,
            i,
            i + len(batch) - 1,
            n_t,
        )
        surface = _load_vol_surface_raw(lake_root, secids=batch, start=start, end=end)
        batch_cube = build_kelly_iv_images(surface, secids=batch, dates=dates_list)
        del surface
        cube[:, i : i + len(batch)] = batch_cube

    if forward_fill and cube.size:
        # ffill along the date axis (axis=0) per (secid, tenor, delta) cell.
        last = np.full(cube.shape[1:], np.nan, dtype=np.float64)
        for i in range(n_t):
            row = cube[i]
            nan_mask = np.isnan(row)
            row = np.where(nan_mask, last, row)
            cube[i] = row
            last = np.where(np.isnan(row), last, row)
    out = cube
    if cache_path is not None:
        import json

        cpath = Path(cache_path)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cpath, cube=out)
        meta_path = Path(str(cpath) + ".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "fingerprint": fp,
                    "shape": list(out.shape),
                    "start": str(start),
                    "end": str(end),
                    "n_secids": n_k,
                    "n_dates": n_t,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _LOG.info("kelly_images cache wrote path=%s shape=%s", cpath, out.shape)
    return out
