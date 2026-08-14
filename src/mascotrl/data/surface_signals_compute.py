"""Surface signal panel assembly and Kelly IV image builders."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from mascotrl.data.surface_signals_extract import (
    _as_cp,
    _grouped_signal_rows,
    _to_frame,
    extract_grid_point,
)
from mascotrl.data.surface_signals_grid import (
    KELLY_DELTAS_CALL,
    KELLY_DELTAS_PUT,
    KELLY_TENORS,
    SURFACE_SIGNAL_NAMES,
    validate_kelly_grid_schema,
    _NAN,
)

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


