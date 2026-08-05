"""Non-trivial baselines for historical PIT OptionMetrics eval (publication).

Literature-grounded ATM-panel peers (CMDP-matched). Soft ruler only.

Existing desk-style:
  - short_vol_carry — static equal short (VRP harvest sleeve; Coval–Shumway /
    Bakshi–Kapadia family)
  - garch_vol_timing — IV vs GARCH(1,1) on ΔIV
  - heston_iv_momentum — short- vs long-horizon IV mean-reversion

Five literature peers:
  - goyal_saretto_hv_iv — Goyal & Saretto (2009) JFE: sort on HV−IV
  - iv_rank_timing — industry IV Rank
  - timed_long_gamma — Bakshi & Kapadia (2003) RFS + desk timing (long only when cheap)
  - skew_risk_reversal — 25Δ put−call skew tilt (Kozhan–Neuberger–Schneider 2013 family)
  - cao_han_high_ivol — Cao & Han (2013) JFE: short high-IVOL optionality

NOT implemented / must not be claimed:
  - Cboe dispersion (short index + long single-name; DSPX/ICJ)
  - Multi-tenor calendars, iron condors, VIX futures, put-write SPX
  - Using ΔATM-IV volatility as “realized vol” (forbidden; HV is stock-return stdev)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from mascotrl.eval.stats_rigor import pack_return_summary
from mascotrl.policy.convex_projection import ConvexProjectionLayer

BASELINE_NAMES = (
    "short_vol_carry",
    "garch_vol_timing",
    "heston_iv_momentum",
    "goyal_saretto_hv_iv",
    "iv_rank_timing",
    "timed_long_gamma",
    "skew_risk_reversal",
    "cao_han_high_ivol",
)

# Peers that require underlier HV (stock-return stdev). Never substitute ΔIV vol.
# cao_han_high_ivol can fall back to demeaned ATM IV level (still not ΔIV-vol).
_HV_REQUIRED = frozenset({"goyal_saretto_hv_iv", "timed_long_gamma"})


def _pack(xs: list[float]) -> dict[str, float]:
    out = pack_return_summary(xs)
    if not xs:
        out["mean_turnover"] = 0.0
    return out


def _garch11_forecast(returns: np.ndarray, *, omega=1e-6, alpha=0.08, beta=0.90) -> np.ndarray:
    """
    Classic GARCH(1,1) variance forecast path (risk-neutral proxy on ATM IV changes).

    σ²_t = ω + α r²_{t-1} + β σ²_{t-1}
    """
    r = np.nan_to_num(np.asarray(returns, dtype=np.float64), nan=0.0)
    n = r.size
    var = np.zeros(n, dtype=np.float64)
    var0 = float(np.var(r[: max(20, n // 10)]) if n > 5 else 1e-4) + 1e-8
    var[0] = var0
    for t in range(1, n):
        var[t] = omega + alpha * (r[t - 1] ** 2) + beta * var[t - 1]
        var[t] = max(var[t], 1e-12)
    return np.sqrt(var)


def rolling_hv_from_returns(
    underlier_rets: np.ndarray,
    *,
    t: int,
    lookback: int = 252,
    min_obs: int = 20,
) -> np.ndarray:
    """Annualized trailing HV from stock returns up to (not including) index ``t``.

    ``underlier_rets`` shape (n_dates, k). Uses only ``rets[:t]`` — no lookahead.
    Returns shape (k,) with NaN where insufficient history.

    CRITICAL: this is stock-return stdev, never ΔIV volatility.
    """
    rets = np.asarray(underlier_rets, dtype=np.float64)
    if rets.ndim != 2:
        raise ValueError("underlier_rets must be (n_dates, k)")
    k = rets.shape[1]
    out = np.full(k, np.nan, dtype=np.float64)
    if t < min_obs:
        return out
    start = max(0, t - int(lookback))
    window = rets[start:t]
    for i in range(k):
        col = window[:, i]
        finite = col[np.isfinite(col)]
        if finite.size < min_obs:
            continue
        out[i] = float(np.std(finite, ddof=0) * np.sqrt(252.0))
    return out


def load_underlier_returns_matrix(
    secids: Sequence[int],
    dates: Sequence,
    *,
    path: Path | str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load OptionMetrics ``secid`` daily stock returns aligned to ``dates``.

    Source: ``TIER_B["sp500_sec"]`` under ``RAW_ROOT/macro``
    (columns: secid, date, return). Never fabricates RV from ΔIV.
    """
    from mascotrl.data.paths import TIER_B

    csv_path = Path(path) if path is not None else TIER_B["sp500_sec"]
    meta: dict[str, Any] = {
        "path": str(csv_path),
        "ok": False,
        "n_secids": len(secids),
        "coverage": 0.0,
        "note": "stock-return HV only; ΔIV vol forbidden as RV proxy",
    }
    n = len(dates)
    k = len(secids)
    out = np.full((n, k), np.nan, dtype=np.float64)
    if k == 0 or n == 0:
        meta["reason"] = "empty secids/dates"
        return out, meta
    if not csv_path.is_file():
        meta["reason"] = f"missing {csv_path}"
        return out, meta

    import duckdb
    import pandas as pd

    id_list = ", ".join(str(int(s)) for s in secids)
    d1 = pd.Timestamp(dates[-1]).strftime("%Y-%m-%d")
    d0_lead = (pd.Timestamp(dates[0]) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    con = duckdb.connect()
    try:
        q = f"""
            SELECT secid, CAST(date AS DATE) AS date, TRY_CAST(return AS DOUBLE) AS ret
            FROM read_csv_auto('{csv_path.as_posix()}', header=true)
            WHERE secid IN ({id_list})
              AND CAST(date AS DATE) BETWEEN DATE '{d0_lead}' AND DATE '{d1}'
            ORDER BY date, secid
        """
        df = con.execute(q).fetchdf()
    finally:
        con.close()
    if df.empty:
        meta["reason"] = "no rows for secids in date range"
        return out, meta

    df["date"] = pd.to_datetime(df["date"])
    slot = {int(s): i for i, s in enumerate(secids)}
    df = df[df["secid"].isin(slot)]
    piv = df.pivot_table(index="date", columns="secid", values="ret", aggfunc="last")
    piv = piv.reindex(columns=[int(s) for s in secids])
    date_index = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    # Include lead-in history so rolling HV at panel start has lookback.
    full_idx = piv.index.union(date_index).sort_values()
    full = piv.reindex(full_idx)
    # Map panel dates into the full array; callers pass panel-length dates only,
    # so we reindex underlier to the *panel* dates after computing on full history
    # is handled by extending dates upstream. Here: align exactly to ``dates``.
    aligned = full.reindex(date_index)
    # If lead-in missing on panel index, try to stitch: for HV we need history
    # *before* panel — so return a longer matrix when lead dates exist.
    # Prefer returning matrix indexed by panel dates but filled via asof from full.
    for col in aligned.columns:
        s = full[col].dropna()
        if s.empty:
            continue
        aligned[col] = s.reindex(date_index, method=None)
    # Better: build matrix on union then slice — actually for rolling HV inside
    # the suite we need underlier_rets aligned 1:1 with atm rows. Prepend lead
    # by extending is done in run_and_attach. Here keep panel-aligned; caller
    # should pass dates that already include lead OR we expand.
    #
    # Expand: return (n_full, k) with dates_full in meta when lead exists.
    if len(full_idx) > len(date_index):
        # Keep panel-length only; lead rows already in piv — merge via reindex
        # of panel dates using values from full (NaN if no exact date).
        out = aligned.to_numpy(dtype=np.float64)
        # Also stash lead-extended array for HV quality if coverage low on early days.
        meta["n_lead_available"] = int(len(full_idx) - len(date_index))
    else:
        out = aligned.to_numpy(dtype=np.float64)

    # Improve coverage: forward-fill is forbidden for returns; leave NaN.
    finite_frac = float(np.isfinite(out).mean()) if out.size else 0.0
    meta["ok"] = finite_frac > 0.05
    meta["coverage"] = finite_frac
    meta["first_date"] = str(date_index[0].date())
    meta["last_date"] = str(date_index[-1].date())
    if not meta["ok"]:
        meta["reason"] = f"coverage too low ({finite_frac:.3f})"
    # Provide lead-extended returns for better HV: (n_lead+n, k) + dates_full
    lead_dates = [d for d in full_idx if d < date_index[0]]
    if lead_dates:
        lead_mat = full.reindex(lead_dates).to_numpy(dtype=np.float64)
        meta["lead_rets"] = lead_mat
        meta["lead_dates"] = [str(pd.Timestamp(d).date()) for d in lead_dates]
    return out, meta


def _xs_zscore(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    m = np.nanmean(arr)
    s = np.nanstd(arr)
    if not np.isfinite(s) or s < 1e-12:
        return np.zeros_like(arr)
    z = (arr - m) / s
    return np.nan_to_num(z, nan=0.0)


def _iv_rank(series: np.ndarray) -> float:
    """Industry IV Rank on a 1-d ATM history ending at the last observation."""
    s = np.asarray(series, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size < 2:
        return float("nan")
    lo = float(np.min(s))
    hi = float(np.max(s))
    cur = float(s[-1])
    if hi - lo < 1e-12:
        return 0.5
    return float((cur - lo) / (hi - lo))


def _baseline_raw_signal(
    name: str,
    *,
    atm_row: np.ndarray,
    atm_hist: np.ndarray,
    deltas: np.ndarray,
    target_gross: float = 1.0,
    skew_row: np.ndarray | None = None,
    hv_row: np.ndarray | None = None,
) -> np.ndarray:
    """Map baseline name → raw K-weights (before CMDP)."""
    k = int(atm_row.shape[0])
    raw = np.zeros(k, dtype=np.float64)
    atm = np.nan_to_num(np.asarray(atm_row, dtype=np.float64), nan=0.2)
    hist = np.asarray(atm_hist, dtype=np.float64)

    if name == "short_vol_carry":
        raw[:] = -target_gross / max(k, 1)

    elif name == "garch_vol_timing":
        for i in range(k):
            series = hist[:, i]
            d_iv = np.diff(series, prepend=series[0])
            fcast = _garch11_forecast(d_iv)
            level_proxy = float(series[-1]) if np.isfinite(series[-1]) else float(atm[i])
            f_vol = float(fcast[-1])
            edge = level_proxy - (float(np.nanmean(series[-20:])) + f_vol)
            raw[i] = -np.tanh(edge * 10.0) * (target_gross / max(k, 1))

    elif name == "heston_iv_momentum":
        for i in range(k):
            series = hist[:, i]
            short = float(np.nanmean(series[-5:]))
            long = (
                float(np.nanmean(series[-20:]))
                if series.size >= 20
                else float(np.nanmean(series))
            )
            raw[i] = -np.tanh((short - long) * 20.0) * (target_gross / max(k, 1))

    elif name == "goyal_saretto_hv_iv":
        # Long vol when HV > IV (Goyal–Saretto 2009).
        if hv_row is None:
            raise KeyError("goyal_saretto_hv_iv requires hv_row (stock-return HV)")
        hv = np.asarray(hv_row, dtype=np.float64)
        for i in range(k):
            if not np.isfinite(hv[i]) or not np.isfinite(atm[i]):
                raw[i] = 0.0
                continue
            raw[i] = np.tanh(8.0 * (float(hv[i]) - float(atm[i]))) * (
                target_gross / max(k, 1)
            )

    elif name == "iv_rank_timing":
        for i in range(k):
            ivr = _iv_rank(hist[:, i])
            if not np.isfinite(ivr):
                raw[i] = 0.0
                continue
            raw[i] = -np.tanh(3.0 * (2.0 * ivr - 1.0)) * (target_gross / max(k, 1))

    elif name == "timed_long_gamma":
        # Long only when cheap (HV > IV); else flat. Bakshi–Kapadia timing.
        if hv_row is None:
            raise KeyError("timed_long_gamma requires hv_row (stock-return HV)")
        hv = np.asarray(hv_row, dtype=np.float64)
        for i in range(k):
            if not np.isfinite(hv[i]) or not np.isfinite(atm[i]):
                raw[i] = 0.0
                continue
            cheap = float(hv[i]) - float(atm[i])
            raw[i] = max(0.0, np.tanh(8.0 * cheap)) * (target_gross / max(k, 1))

    elif name == "skew_risk_reversal":
        if skew_row is None:
            raise KeyError("skew_risk_reversal requires skew_row")
        z = _xs_zscore(np.asarray(skew_row, dtype=np.float64))
        raw = -np.tanh(z) * (target_gross / max(k, 1))

    elif name == "cao_han_high_ivol":
        # Short high-IVOL names (Cao–Han 2013 proxy via trailing HV).
        if hv_row is not None and np.isfinite(hv_row).any():
            z = _xs_zscore(np.asarray(hv_row, dtype=np.float64))
        else:
            # Fallback: demeaned ATM IV level (still not ΔIV-vol-as-RV).
            z = _xs_zscore(atm)
        raw = -np.tanh(z) * (target_gross / max(k, 1))

    else:
        raise KeyError(name)

    # Soft delta-neutral tilt: shrink names with large |Δ|.
    d = np.nan_to_num(np.asarray(deltas, dtype=np.float64), nan=0.0)
    shrink = 1.0 / (1.0 + 2.0 * np.abs(d))
    raw = raw * shrink
    if np.abs(raw).sum() > 1e-12:
        raw = raw * (target_gross / (np.abs(raw).sum() + 1e-12))
    else:
        raw = np.zeros(k, dtype=np.float64)
    return raw


def baseline_weights_day(
    name: str,
    *,
    atm_row: np.ndarray,
    atm_hist: np.ndarray,
    deltas: np.ndarray,
    w_prev: torch.Tensor,
    projector: ConvexProjectionLayer,
    vol_scale: float,
    target_gross: float = 1.0,
    skew_row: np.ndarray | None = None,
    hv_row: np.ndarray | None = None,
) -> torch.Tensor:
    """Map baseline name → proposed weights, then CMDP-project for fair turnover/δ."""
    raw = _baseline_raw_signal(
        name,
        atm_row=atm_row,
        atm_hist=atm_hist,
        deltas=deltas,
        target_gross=target_gross,
        skew_row=skew_row,
        hv_row=hv_row,
    )
    d = np.nan_to_num(np.asarray(deltas, dtype=np.float64), nan=0.0)
    w_tgt = torch.from_numpy(raw.astype(np.float32)).view(1, -1)
    return projector(
        w_tgt,
        w_prev,
        torch.from_numpy(d.astype(np.float32)).view(1, -1),
        vol_scale=vol_scale,
    )


def _baseline_raw_weights(
    name: str,
    *,
    atm_row: np.ndarray,
    atm_hist: np.ndarray,
    deltas: np.ndarray,
    target_gross: float = 1.0,
    skew_row: np.ndarray | None = None,
    hv_row: np.ndarray | None = None,
) -> torch.Tensor:
    """Same signal map as ``baseline_weights_day`` but without CMDP projection."""
    raw = _baseline_raw_signal(
        name,
        atm_row=atm_row,
        atm_hist=atm_hist,
        deltas=deltas,
        target_gross=target_gross,
        skew_row=skew_row,
        hv_row=hv_row,
    )
    return torch.from_numpy(raw.astype(np.float32)).view(1, -1)


def run_baseline_suite_on_panel(
    *,
    atm: np.ndarray,
    deltas_np: np.ndarray,
    fwd: np.ndarray,
    dates: list,
    seq_len: int,
    turnover_limit: float = 0.15,
    target_gross: float = 1.0,
    ledger: Any | None = None,
    phase: str = "OOS_TEST",
    use_projection: bool = True,
    max_name_abs_weight: float = 5.0,
    skew: np.ndarray | None = None,
    underlier_rets: np.ndarray | None = None,
    underlier_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Walk the same panel as hist OOS; return pnls/summary for each baseline.

    When ``ledger`` is set, records per-ticker executed weights (strategy=baseline
    name) for academic comparison against HAPPO allocations.

    ``use_projection`` must match the HAPPO arm under comparison (False on −CMDP).

    ``underlier_rets`` must be stock returns (secid), never ΔIV. HV peers are
    skipped (flat PnL + unavailable reason) if HV cannot be formed.
    """
    n, k = atm.shape
    seq = max(4, int(seq_len))
    projector = None
    if use_projection:
        projector = ConvexProjectionLayer(
            num_assets=k,
            turnover_limit=turnover_limit,
            max_name_abs_weight=max_name_abs_weight,
        )
    modes = BASELINE_NAMES
    pnls: dict[str, list[float]] = {m: [] for m in modes}
    turns: dict[str, list[float]] = {m: [] for m in modes}
    dates_used: list[str] = []
    w_prev = {m: torch.zeros(1, k) for m in modes}

    # Optionally prepend lead returns so early-panel HV has 252d history.
    urets = None
    u_offset = 0
    if underlier_rets is not None:
        urets = np.asarray(underlier_rets, dtype=np.float64)
        lead = (underlier_meta or {}).get("lead_rets")
        if lead is not None:
            lead_arr = np.asarray(lead, dtype=np.float64)
            if lead_arr.ndim == 2 and lead_arr.shape[1] == k:
                urets = np.vstack([lead_arr, urets])
                u_offset = int(lead_arr.shape[0])
    hv_ok = urets is not None and urets.shape[1] == k and urets.shape[0] >= n
    skew_ok = skew is not None and np.asarray(skew).shape == (n, k)
    skipped: dict[str, str] = {}
    if not hv_ok:
        for m in _HV_REQUIRED:
            skipped[m] = (
                "underlier stock-return HV unavailable "
                "(ΔIV-vol forbidden as substitute)"
            )
    if not skew_ok:
        skipped["skew_risk_reversal"] = "skew_25d matrix unavailable"

    with torch.no_grad():
        for t in range(seq, n):
            hist = np.nan_to_num(atm[t - seq : t], nan=0.0)
            drow = np.nan_to_num(deltas_np[t - 1], nan=0.0)
            ret = np.nan_to_num(fwd[t - 1], nan=0.0).astype(np.float32)
            vol_scale = float(np.nanmean(atm[t - 1]))
            if not np.isfinite(vol_scale) or vol_scale <= 0:
                vol_scale = 0.2
            skew_row = (
                np.nan_to_num(skew[t - 1], nan=0.0) if skew_ok else None  # type: ignore[index]
            )
            hv_row = (
                rolling_hv_from_returns(urets, t=u_offset + t - 1)
                if hv_ok
                else None
            )
            day = __import__("pandas").Timestamp(dates[t])
            for m in modes:
                if m in skipped:
                    pnls[m].append(0.0)
                    turns[m].append(0.0)
                    w_prev[m] = torch.zeros(1, k)
                    continue
                try:
                    if projector is not None:
                        w = baseline_weights_day(
                            m,
                            atm_row=np.nan_to_num(atm[t - 1], nan=0.0),
                            atm_hist=hist,
                            deltas=drow,
                            w_prev=w_prev[m],
                            projector=projector,
                            vol_scale=vol_scale,
                            target_gross=target_gross,
                            skew_row=skew_row,
                            hv_row=hv_row,
                        )
                    else:
                        w = _baseline_raw_weights(
                            m,
                            atm_row=np.nan_to_num(atm[t - 1], nan=0.0),
                            atm_hist=hist,
                            deltas=drow,
                            target_gross=target_gross,
                            skew_row=skew_row,
                            hv_row=hv_row,
                        )
                except KeyError as exc:
                    skipped[m] = str(exc)
                    pnls[m].append(0.0)
                    turns[m].append(0.0)
                    continue
                w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
                pnl = float((w.squeeze(0).numpy() * ret).sum())
                if not np.isfinite(pnl):
                    pnl = 0.0
                turn = float((w - w_prev[m]).abs().sum().item())
                if not np.isfinite(turn):
                    turn = 0.0
                pnls[m].append(pnl)
                turns[m].append(turn)
                if ledger is not None:
                    ledger.record_step(
                        date=day,
                        phase=phase,
                        weights_exec=w,
                        deltas=drow,
                        step_pnl=pnl,
                        step=t,
                        strategy=m,
                    )
                w_prev[m] = w.detach()
            dates_used.append(str(day.date()))

    summary = {m: _pack(pnls[m]) for m in modes}
    for m in modes:
        summary[m]["mean_turnover"] = float(np.mean(turns[m])) if turns[m] else 0.0
        if m in skipped:
            summary[m]["unavailable"] = True
            summary[m]["reason"] = skipped[m]
    umeta = dict(underlier_meta or {})
    umeta.pop("lead_rets", None)
    if isinstance(umeta.get("lead_dates"), list) and len(umeta["lead_dates"]) > 20:
        umeta["n_lead_dates"] = len(umeta["lead_dates"])
        umeta.pop("lead_dates", None)
    return {
        "protocol": "pit_optionmetrics_baselines",
        "modes": list(modes),
        "summary": summary,
        "pnls": pnls,
        "turnovers": turns,
        "dates": dates_used,
        "n_days": len(dates_used),
        "first_date": dates_used[0] if dates_used else None,
        "last_date": dates_used[-1] if dates_used else None,
        "use_projection": bool(use_projection),
        "skipped": skipped,
        "underlier_meta": umeta,
        "citations": {
            "goyal_saretto_hv_iv": "Goyal & Saretto (2009) JFE",
            "iv_rank_timing": "Industry IV Rank",
            "timed_long_gamma": "Bakshi & Kapadia (2003) RFS + desk timing",
            "skew_risk_reversal": "25Δ RR; Kozhan–Neuberger–Schneider (2013) RFS family",
            "cao_han_high_ivol": "Cao & Han (2013) JFE",
        },
        "non_claims": [
            "Not Cboe dispersion / DSPX / ICJ",
            "Not multi-tenor calendars or iron condors",
            "Not VIX futures / put-write / SPX overlay",
            "HV is stock-return stdev — never ΔIV vol",
        ],
    }
