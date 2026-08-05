"""Market/cash orientation benches — scalar rivals, never via BASELINE_NAMES.

Equity: lake ``macro/sp500_prices.parquet`` (prefer ``vwretd``, else ``sprtrn``).
Cash: lake ``macro/interest_rate.parquet`` — EFFR then DTB3 levels → daily
``r = (rate/100)/252``. SOFR is not the sole RF for 2007–2021 panels.

These are calendar-aligned Sharpe anchors for non-ML readability. They are
**not** the same economic object as ATM-mid option-book PnL and must not be
fed through the K-weight + CMDP baseline API.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from mascotrl.data.paths import LAKE_ROOT
from mascotrl.eval.baselines import BASELINE_NAMES
from mascotrl.eval.stats_rigor import pack_return_summary
from mascotrl.logging_utils import get_logger

log = get_logger("mascotrl.eval.orientation")

ORIENTATION_NAMES = ("cash_rf", "equity_market")
UNIT_DISCLAIMER = (
    "Orientation: equity Sharpe and cash annualized mean on the hist calendar — "
    "not dollar overlays of the ATM-mid option book. Cash RF is near-deterministic, "
    "so cash Sharpe is omitted (NaN); use cash_rf_mean_ann."
)


def _pack(xs: list[float] | np.ndarray, *, near_deterministic: bool = False) -> dict[str, float]:
    arr = np.asarray(xs, dtype=np.float64)
    finite = np.isfinite(arr)
    use = arr[finite] if finite.any() else arr
    if use.size == 0:
        out = pack_return_summary([])
        out["mean_ann"] = float("nan")
        out["pnl_sum"] = float("nan")
        return out
    out = pack_return_summary(xs)
    mu = float(use.mean())
    sd = float(use.std(ddof=0))
    out["mean_ann"] = float(mu * 252.0)
    # Cash RF daily rates are nearly constant → Sharpe is not a useful statistic.
    if near_deterministic or sd < 1e-8:
        out["sharpe"] = float("nan")
    return out


def load_equity_daily_returns(
    lake_base_dir: Path | str | None = None,
) -> pd.Series:
    """One return per calendar date from CRSP-style lake prices."""
    lake = Path(lake_base_dir) if lake_base_dir else LAKE_ROOT
    path = lake / "macro" / "sp500_prices.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"equity lake missing: {path}")
    df = pd.read_parquet(path, columns=["date", "vwretd", "sprtrn"])
    df["date"] = pd.to_datetime(df["date"])
    # Prefer value-weighted market return; fall back to S&P return.
    if df["vwretd"].notna().any():
        col = "vwretd"
    elif df["sprtrn"].notna().any():
        col = "sprtrn"
    else:
        raise RuntimeError(f"no vwretd/sprtrn in {path}")
    daily = (
        df.groupby("date", sort=True)[col]
        .first()
        .astype(np.float64)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    daily.name = col
    return daily


def load_cash_daily_returns(
    lake_base_dir: Path | str | None = None,
) -> pd.Series:
    """Overnight cash proxy from EFFR (preferred) or DTB3 level → daily rate."""
    lake = Path(lake_base_dir) if lake_base_dir else LAKE_ROOT
    path = lake / "macro" / "interest_rate.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"rates lake missing: {path}")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    rate_col = None
    for c in ("effr", "dtb3", "dff"):
        if c in df.columns and df[c].notna().any():
            rate_col = c
            break
    if rate_col is None:
        raise RuntimeError(f"no effr/dtb3/dff in {path}")
    level = (
        df.groupby("date", sort=True)[rate_col]
        .first()
        .astype(np.float64)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    # Annual percent → daily simple return.
    daily = (level / 100.0) / 252.0
    daily.name = f"cash_{rate_col}"
    return daily


def align_to_dates(
    series: pd.Series,
    dates: Sequence[Any],
) -> tuple[np.ndarray, list[str], float]:
    """Align a daily series to hist dates; return values, used dates, coverage."""
    idx = pd.to_datetime(pd.Index(list(dates)))
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    aligned = s.reindex(idx)
    finite = aligned.to_numpy(dtype=np.float64)
    n_ok = int(np.isfinite(finite).sum())
    coverage = float(n_ok / max(len(idx), 1))
    used = [d.strftime("%Y-%m-%d") for d in idx]
    return finite, used, coverage


def run_orientation_benchmarks(
    dates: Sequence[Any],
    *,
    lake_base_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Compute cash_rf + equity_market summary on the hist calendar."""
    assert "cash_rf" not in BASELINE_NAMES
    assert "equity_market" not in BASELINE_NAMES

    out: dict[str, Any] = {
        "protocol": "orientation_scalar_rivals",
        "unit_disclaimer": UNIT_DISCLAIMER,
        "summary": {},
        "pnls": {},
        "source": {},
        "coverage": {},
    }
    if not dates:
        out["status"] = "empty_dates"
        return out

    try:
        equity = load_equity_daily_returns(lake_base_dir)
        eq_vals, _, eq_cov = align_to_dates(equity, dates)
        out["pnls"]["equity_market"] = [float(x) if np.isfinite(x) else float("nan") for x in eq_vals]
        out["summary"]["equity_market"] = _pack(eq_vals)
        out["source"]["equity_market"] = str(equity.name)
        out["coverage"]["equity_market"] = eq_cov
    except Exception as exc:
        log.warning("orientation equity unavailable: %s", exc)
        out["summary"]["equity_market"] = _pack([])
        out["pnls"]["equity_market"] = []
        out["source"]["equity_market"] = None
        out["coverage"]["equity_market"] = 0.0
        out["equity_error"] = str(exc)

    try:
        cash = load_cash_daily_returns(lake_base_dir)
        c_vals, _, c_cov = align_to_dates(cash, dates)
        out["pnls"]["cash_rf"] = [float(x) if np.isfinite(x) else float("nan") for x in c_vals]
        out["summary"]["cash_rf"] = _pack(c_vals, near_deterministic=True)
        out["source"]["cash_rf"] = str(cash.name)
        out["coverage"]["cash_rf"] = c_cov
    except Exception as exc:
        log.warning("orientation cash unavailable: %s", exc)
        out["summary"]["cash_rf"] = _pack([])
        out["pnls"]["cash_rf"] = []
        out["source"]["cash_rf"] = None
        out["coverage"]["cash_rf"] = 0.0
        out["cash_error"] = str(exc)

    out["n_dates"] = len(dates)
    out["status"] = "ok"
    return out


def run_and_attach_orientation_benchmarks(
    report: dict[str, Any],
    *,
    lake_base_dir: Path | str | None = None,
    dates: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Attach ``orientation_benchmarks`` from hist OOS dates (or explicit dates)."""
    if dates is None:
        dates = (report.get("historical_oos") or {}).get("dates") or []
        if not dates:
            cal = report.get("historical_calendar") or {}
            dates = cal.get("oos_dates") or []
    suite = run_orientation_benchmarks(dates, lake_base_dir=lake_base_dir)
    report["orientation_benchmarks"] = suite

    happo = (
        ((report.get("historical_oos") or {}).get("summary") or {}).get("happo") or {}
    )
    happo_sh = float(happo.get("sharpe", float("nan")))
    lead = {
        "happo_sharpe": happo_sh,
        "cash_rf_sharpe": float(
            (suite.get("summary") or {}).get("cash_rf", {}).get("sharpe", float("nan"))
        ),
        "cash_rf_mean_ann": float(
            (suite.get("summary") or {}).get("cash_rf", {}).get("mean_ann", float("nan"))
        ),
        "equity_market_sharpe": float(
            (suite.get("summary") or {})
            .get("equity_market", {})
            .get("sharpe", float("nan"))
        ),
        "unit_disclaimer": UNIT_DISCLAIMER,
        "friction_applied": bool(
            (report.get("historical_oos") or {}).get("friction_applied", False)
        ),
    }
    if np.isfinite(lead["happo_sharpe"]) and np.isfinite(lead["cash_rf_sharpe"]):
        lead["happo_minus_cash"] = lead["happo_sharpe"] - lead["cash_rf_sharpe"]
    if np.isfinite(lead["happo_sharpe"]) and np.isfinite(lead["equity_market_sharpe"]):
        lead["happo_minus_equity"] = lead["happo_sharpe"] - lead["equity_market_sharpe"]
    report["orientation_lead"] = lead
    log.info(
        "orientation benches happo=%.3f cash=%.3f equity=%.3f coverage_eq=%.2f",
        lead["happo_sharpe"],
        lead["cash_rf_sharpe"],
        lead["equity_market_sharpe"],
        float((suite.get("coverage") or {}).get("equity_market", 0.0)),
    )
    return report
