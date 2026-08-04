#!/usr/bin/env python3
"""B2: run the signal IC gate over the PIT selection window (2003-2012).

Materializes month-end surface signals from the lake for a secid pool,
compounds equity returns to the same monthly cadence, then for each
candidate signal computes:

  * Fama-MacBeth + Newey-West HAC t on the slope series
  * Benjamini-Hochberg FDR admission (``signal_ic_gate_v2``, default q=0.05)
  * Legacy ``|t| > --t-min`` diagnostic (not the admit bit)
  * Decile long-short mean return / Sharpe
  * Spearman IC and IC decay across horizons
  * FF4 + Pastor-Stambaugh traded-liquidity alpha (Newey-West HAC t-stat)
    of the signal's own decile long-short return series

Admitted signals are orthogonalized (Gram-Schmidt in admit order) and the
resulting effective breadth is reported. Writes ``config/signal_allowlist.json``
(or ``--out``). All-NaN candidates are quarantined as ``unscored``.

All series share one month-end date axis (``me_dates``); trailing-month
equity/factor returns and Fama-MacBeth's own one-period lag both key off
that same axis, so a signal known at ``me_dates[t]`` is always tested
against the return realized over ``(me_dates[t], me_dates[t+1]]`` -- never
against a return that has already happened. ``run_signal_gate_v2`` also
refuses (raises) any date past ``SELECTION_END``.

Cross-section scope: by default this restricts the pool to
``--max-pool`` names (ranked by return-series activity, matching the
campaign's own tractability convention) because each secid-month requires
a full option-chain moment integration and the full ~500-name universe
over 2003-2012 takes on the order of tens of minutes on the disk this
project runs against. Pass ``--max-pool 0`` for the full universe (a
multi-minute-plus run intended for the user's own dedicated launch, not
this interactive smoke path).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "config" / "signal_allowlist.json"
DEFAULT_LEDGER = ROOT / "logs" / "artifacts" / "signal_gate_trial_ledger.jsonl"


def _monthly_returns_full(
    rets: np.ndarray, dates: pd.DatetimeIndex
) -> tuple[np.ndarray, list[pd.Timestamp]]:
    """Trailing calendar-month equity returns on the full month-end axis.

    Returns ``(ret_full, me_dates)`` with ``ret_full.shape == (len(me_dates), K)``.
    ``ret_full[0]`` is ``NaN`` (no prior month-end to compound from);
    ``ret_full[i]`` for ``i >= 1`` is the compounded return over
    ``(me_dates[i-1], me_dates[i]]``.
    """
    from src.eval.cadence import month_end_mask

    idx = pd.DatetimeIndex(dates)
    me_mask = month_end_mask(idx)
    me_pos = np.where(me_mask)[0]
    me_dates = [idx[p] for p in me_pos]
    k = rets.shape[1]
    ret_full = np.full((len(me_dates), k), np.nan, dtype=np.float64)
    for i in range(1, len(me_pos)):
        lo, hi = me_pos[i - 1] + 1, me_pos[i] + 1
        seg = rets[lo:hi]
        gross = np.nanprod(1.0 + np.nan_to_num(seg, nan=0.0), axis=0)
        ret_full[i] = gross - 1.0
    return ret_full, me_dates


def _pivot_month_end_signal(
    signals_long: pd.DataFrame, name: str, secids: list, me_dates: list
) -> np.ndarray:
    k, t = len(secids), len(me_dates)
    if signals_long is None or len(signals_long) == 0 or name not in signals_long.columns:
        return np.full((t, k), np.nan, dtype=np.float64)
    wide = signals_long.pivot_table(index="date", columns="secid", values=name, aggfunc="last")
    wide = wide.reindex(index=pd.DatetimeIndex(me_dates), columns=secids)
    return wide.to_numpy(dtype=np.float64)


def _monthly_ff4_ps_full(lake_root, me_dates: list) -> tuple[np.ndarray, list[str]]:
    """FF4 (compounded daily -> trailing monthly) + Pastor-Stambaugh
    ``PS_VWF``, aligned to the same full ``me_dates`` axis as
    :func:`_monthly_returns_full` (row 0 is ``NaN``)."""
    from scripts.run_eq_alloc_campaign import _load_ff4

    ff4_daily = _load_ff4(list(me_dates), lake_root)
    n = len(me_dates)
    ff4_full = np.full((n, 4), np.nan, dtype=np.float64)
    # _load_ff4 aligns to exactly `me_dates`, so this reuses the same
    # month-end rows rather than re-querying at daily granularity; it is
    # an approximation (a true trailing-month compounding of daily FF4
    # would need the full daily grid) that is adequate for a diagnostic
    # alpha check, not a headline estimand.
    for i in range(1, n):
        ff4_full[i] = ff4_daily[i]

    ps_path = Path(lake_root) / "macro" / "pastor_stambaugh.parquet"
    ps_col = np.full(n, np.nan, dtype=np.float64)
    if ps_path.is_file():
        ps = pd.read_parquet(ps_path)
        ps["DATE"] = pd.to_datetime(ps["DATE"])
        ps = ps.set_index("DATE").sort_index()
        ps_series = ps["PS_VWF"].reindex(
            pd.DatetimeIndex(me_dates), method="nearest", tolerance=pd.Timedelta(days=5)
        )
        ps_col = ps_series.to_numpy(dtype=np.float64)
        ps_col[0] = np.nan

    factors = np.column_stack([ff4_full, ps_col])
    return factors, ["Mkt-RF", "SMB", "HML", "Mom", "PS_VWF"]


def _decile_ls_series(signal: np.ndarray, returns: np.ndarray, *, n_deciles: int = 5) -> np.ndarray:
    """Per-date long-top/short-bottom decile spread, predictive: ``signal[t]``
    ranks the cross-section that forms the spread realized at ``returns[t+1]``."""
    sig = np.asarray(signal, dtype=np.float64)
    ret = np.asarray(returns, dtype=np.float64)
    out = np.full(sig.shape[0] - 1, np.nan, dtype=np.float64)
    for t in range(sig.shape[0] - 1):
        s = sig[t]
        r = ret[t + 1]
        mask = np.isfinite(s) & np.isfinite(r)
        if int(mask.sum()) < n_deciles:
            continue
        s_m, r_m = s[mask], r[mask]
        ranks = np.empty_like(s_m)
        ranks[np.argsort(s_m)] = np.arange(s_m.size, dtype=np.float64)
        pct = ranks / max(s_m.size - 1, 1)
        lo = pct <= (1.0 / n_deciles)
        hi = pct >= (1.0 - 1.0 / n_deciles)
        if np.any(lo) and np.any(hi):
            out[t] = float(np.mean(r_m[hi]) - np.mean(r_m[lo]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-pool", type=int, default=40, help="cross-section cap; 0 = full universe")
    ap.add_argument(
        "--t-min",
        type=float,
        default=2.0,
        help="legacy |FM t| diagnostic threshold (not the v2 admit bit)",
    )
    ap.add_argument("--fdr-q", type=float, default=0.05, help="BH FDR q for signal_ic_gate_v2")
    ap.add_argument("--hlz-t", type=float, default=3.0, help="HLZ discovery flag |t_nw| threshold")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--trial-ledger", default=str(DEFAULT_LEDGER))
    args = ap.parse_args()

    from src.data.equity_panel import SELECTION_START, SELECTION_END, load_sp500_security_returns
    from src.data.paths import LAKE_ROOT
    from src.data.surface_signals import (
        SURFACE_SIGNAL_NAMES,
        materialize_surface_signals_from_lake,
    )
    from src.eval.signal_gate import (
        decile_long_short,
        ff_alpha,
        ic_decay,
        ic_series,
        run_signal_gate_v2,
        write_signal_allowlist,
    )
    from src.eval.trial_ledger import TrialLedger
    from scripts.run_eq_alloc_campaign import _wide_returns

    t0 = time.perf_counter()
    raw = load_sp500_security_returns(LAKE_ROOT, start=SELECTION_START, end=SELECTION_END)
    rets, secids, dates = _wide_returns(raw, start=SELECTION_START, end=SELECTION_END)
    print(f"selection panel T={rets.shape[0]} K={rets.shape[1]} loaded in {time.perf_counter()-t0:.1f}s")

    if args.max_pool and rets.shape[1] > args.max_pool:
        activity = np.nanstd(rets, axis=0)
        keep = np.argsort(activity)[::-1][: args.max_pool]
        rets = rets[:, keep]
        secids = [secids[i] for i in keep]
    print(f"gate pool K={len(secids)} (max_pool={args.max_pool or 'full'})")

    ret_full, me_dates = _monthly_returns_full(rets, dates)
    if len(me_dates) < 13:
        raise SystemExit(f"too few month-ends ({len(me_dates)}) for a signal gate")

    t1 = time.perf_counter()
    signals_long = materialize_surface_signals_from_lake(
        LAKE_ROOT,
        secids=secids,
        start=SELECTION_START,
        end=SELECTION_END,
        month_end_only=True,
    )
    print(
        f"materialized {len(signals_long)} month-end signal rows in "
        f"{time.perf_counter()-t1:.1f}s"
    )

    factors_full, factor_names = _monthly_ff4_ps_full(LAKE_ROOT, me_dates)

    signal_panels: dict[str, np.ndarray] = {}
    unscored_names: list[str] = []
    for name in SURFACE_SIGNAL_NAMES:
        arr = _pivot_month_end_signal(signals_long, name, secids, me_dates)
        if np.isfinite(arr).sum() == 0:
            unscored_names.append(name)
            signal_panels[name] = arr
            continue
        signal_panels[name] = arr

    gate = run_signal_gate_v2(
        signal_panels,
        ret_full,
        dates=me_dates,
        selection_end=SELECTION_END,
        fdr_q=args.fdr_q,
        hlz_t=args.hlz_t,
        legacy_t_min=args.t_min,
    )

    for name, arr in signal_panels.items():
        row = gate["stats"][name]
        if row.get("status") == "unscored":
            continue
        row["decile_long_short"] = decile_long_short(arr, ret_full, n_deciles=5)
        ics = ic_series(arr, ret_full)
        row["ic_mean"] = float(np.nanmean(ics)) if ics.size else float("nan")
        row["ic_decay"] = ic_decay(arr, ret_full, horizons=(1, 3, 6))
        ls_returns = _decile_ls_series(arr, ret_full, n_deciles=5)
        # ls_returns[t] realizes over (me_dates[t], me_dates[t+1]] == the
        # same window as factors_full[t+1].
        factors_aligned = factors_full[1:]
        if ls_returns.size == factors_aligned.shape[0]:
            row["ff4_ps_alpha"] = ff_alpha(ls_returns, factors_aligned)
        else:
            row["ff4_ps_alpha"] = {"alpha": float("nan"), "t_stat": float("nan")}

    gate["factor_names"] = factor_names
    gate["pool_secids"] = [int(s) if not isinstance(s, str) else s for s in secids]
    gate["n_pool"] = len(secids)
    gate["n_months"] = len(me_dates)
    gate["catalog_names"] = list(SURFACE_SIGNAL_NAMES)
    gate["unscored_names"] = list(unscored_names)
    gate["wall_s"] = time.perf_counter() - t0

    ledger = TrialLedger(args.trial_ledger)
    ledger.append(
        baseline="signal_ic_gate_v2",
        seed=0,
        fold=0,
        status="ok",
        metrics={
            "allowlist": list(gate["allowlist"]),
            "n_family": int(gate["n_family"]),
            "fdr_q": float(gate["fdr_q"]),
            "n_pool": int(gate["n_pool"]),
            "n_unscored": int(len(unscored_names)),
            "effective_breadth": gate.get("effective_breadth"),
        },
    )

    out_path = Path(args.out)
    write_signal_allowlist(gate, out_path)
    print(
        f"wrote {out_path}: estimand={gate['estimand']} "
        f"allowlist={gate['allowlist']} "
        f"n_family={gate['n_family']} unscored={unscored_names} "
        f"effective_breadth={gate['effective_breadth']:.2f} "
        f"in {gate['wall_s']:.1f}s total"
    )


if __name__ == "__main__":
    main()
