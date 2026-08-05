"""D2: persist per-strategy OOS weights/turnover/cost/gross to parquet.

The eq allocation campaign (``scripts/run_eq_alloc_campaign.py``) scores every
peer (benchmark, OLPS, ceiling arm) and the policy through the same parity
harness (``src/eval/parity_harness.score_strategy``), which already returns
per-date weights/turnover/cost/gross/total_net/residual arrays. Those arrays
were discarded after the summary Sharpe was extracted; this module gives them
a durable on-disk home (one parquet per strategy plus one combined
``PortfolioAccountingLedger``) so the reporting book (D3/D4) can render
holdings, turnover, and cost figures without re-running the CPCV roll.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from mascotrl.reporting.portfolio_accounting import PortfolioAccountingLedger
from mascotrl.reporting.book_style import FAMILY_ORDER, strategy_family


def strategy_frame(
    *,
    dates: Sequence[Any],
    secids: Sequence[Any],
    weights: np.ndarray,
    turnover: Sequence[float] | np.ndarray,
    cost: Sequence[float] | np.ndarray,
    gross: Sequence[float] | np.ndarray,
    total_net: Sequence[float] | np.ndarray | None = None,
    residual: Sequence[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    """One strategy's OOS holdings + cost ladder, one row per scored date.

    ``weights`` is ``(n_dates, K)``; per-asset columns are named ``w_<secid>``
    so the frame round-trips through parquet without a MultiIndex.
    """
    w = np.atleast_2d(np.asarray(weights, dtype=np.float64))
    n = w.shape[0]
    cols = ["date", "turnover", "cost", "gross", "total_net", "residual"]
    if n == 0:
        return pd.DataFrame(columns=cols)
    if len(dates) != n:
        raise ValueError(f"dates length {len(dates)} != weights rows {n}")

    def _series(x: Sequence[float] | np.ndarray | None) -> np.ndarray:
        if x is None:
            return np.full(n, np.nan, dtype=np.float64)
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if arr.size != n:
            raise ValueError(f"series length {arr.size} != {n}")
        return arr

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(list(dates)),
            "turnover": _series(turnover),
            "cost": _series(cost),
            "gross": _series(gross),
            "total_net": _series(total_net),
            "residual": _series(residual),
        }
    )
    for j, sid in enumerate(secids):
        if j < w.shape[1]:
            df[f"w_{sid}"] = w[:, j]
    return df


def persist_strategy_frames(
    frames: Mapping[str, pd.DataFrame], out_dir: str | Path
) -> dict[str, str]:
    """Write one parquet file per non-empty strategy frame; return the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, df in frames.items():
        if df is None or df.empty:
            continue
        safe = str(name).replace(":", "__").replace("/", "__")
        path = out_dir / f"{safe}.parquet"
        df.to_parquet(path, index=False)
        written[name] = str(path)
    return written


def build_accounting_ledger(
    *,
    secids: Sequence[Any],
    frames: Mapping[str, pd.DataFrame],
    phase: str = "OOS_TEST",
) -> PortfolioAccountingLedger:
    """One :class:`PortfolioAccountingLedger` spanning every strategy's weights.

    Reuses the ledger's existing multi-strategy ``strategy`` column rather
    than inventing a second forensic-audit format.
    """
    secids = list(secids)
    ledger = PortfolioAccountingLedger(
        asset_names=[str(s) for s in secids], num_assets=len(secids)
    )
    w_cols = [f"w_{sid}" for sid in secids]
    for name, df in frames.items():
        if df is None or df.empty:
            continue
        have_cols = [c for c in w_cols if c in df.columns]
        if not have_cols:
            continue
        w_full = np.zeros((len(df), len(secids)), dtype=np.float64)
        for j, c in enumerate(w_cols):
            if c in df.columns:
                w_full[:, j] = df[c].to_numpy(dtype=np.float64)
        has_pnl = "total_net" in df.columns
        dates_col = df["date"].to_numpy()
        pnl_col = df["total_net"].to_numpy(dtype=np.float64) if has_pnl else None
        for i in range(len(df)):
            ledger.record_step(
                date=dates_col[i],
                phase=phase,
                weights_exec=w_full[i],
                step_pnl=float(pnl_col[i]) if pnl_col is not None and np.isfinite(pnl_col[i]) else 0.0,
                step=i,
                strategy=str(name),
            )
    return ledger


def write_holdings_book(
    *,
    frames: Mapping[str, pd.DataFrame],
    secids: Sequence[Any],
    out_xlsx: str | Path,
    out_csv: str | Path | None = None,
    ticker_by_secid: Mapping[Any, str] | None = None,
    rebalance_turnover_floor: float = 1e-9,
) -> dict[str, str]:
    """Write a human-readable holdings book (Excel + optional CSV twin).

    Sheets: ``Rebalances`` (turnover days only), ``AllDays``, ``Trades``
    (full Δw blotter), ``Turnover``, ``Summary``. Tickers prefer
    ``ticker_by_secid``; fall back to the numeric secid string.
    """
    out_xlsx = Path(out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    name_of = {sid: str((ticker_by_secid or {}).get(sid, sid)) for sid in secids}

    # Wide policy rebalance book (primary sheet source).
    focus = frames.get("policy")
    if focus is None:
        # Fall back to first frame with weights.
        for df in frames.values():
            if df is not None and not df.empty and any(
                str(c).startswith("w_") for c in df.columns
            ):
                focus = df
                break
    if focus is None or focus.empty:
        raise ValueError("write_holdings_book: no strategy frames with weights")

    w_cols = [f"w_{sid}" for sid in secids if f"w_{sid}" in focus.columns]
    rename = {f"w_{sid}": name_of[sid] for sid in secids if f"w_{sid}" in focus.columns}
    all_days = focus[["date"] + w_cols].copy() if "date" in focus.columns else focus[w_cols].copy()
    all_days = all_days.rename(columns=rename)
    if "turnover" in focus.columns and "date" in focus.columns:
        reb = focus.loc[
            focus["turnover"].to_numpy(dtype=float) > float(rebalance_turnover_floor),
            ["date"] + w_cols,
        ].rename(columns=rename)
    else:
        reb = all_days

    # Full trade blotter: day-over-day Δw for every name.
    trade_rows: list[dict[str, Any]] = []
    if "date" in focus.columns and w_cols:
        w_mat = focus[w_cols].to_numpy(dtype=float)
        dates = pd.to_datetime(focus["date"])
        for i in range(1, len(focus)):
            dw = w_mat[i] - w_mat[i - 1]
            for j, c in enumerate(w_cols):
                if abs(float(dw[j])) <= float(rebalance_turnover_floor):
                    continue
                sid = c[2:] if c.startswith("w_") else c
                trade_rows.append(
                    {
                        "date": dates.iloc[i],
                        "ticker": name_of.get(
                            int(sid) if str(sid).isdigit() else sid, str(sid)
                        ),
                        "delta_w": float(dw[j]),
                        "w_prev": float(w_mat[i - 1, j]),
                        "w_new": float(w_mat[i, j]),
                    }
                )
    trades = pd.DataFrame(trade_rows)

    turnover = (
        focus[["date", "turnover"]].copy()
        if "turnover" in focus.columns and "date" in focus.columns
        else pd.DataFrame()
    )
    summary_rows = []
    for name, df in frames.items():
        if df is None or df.empty:
            continue
        row: dict[str, Any] = {"strategy": name, "n_days": len(df)}
        if "total_net" in df.columns:
            x = df["total_net"].to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if x.size >= 2:
                sd = float(np.std(x, ddof=1))
                row["sharpe"] = (
                    float(np.sqrt(252.0) * float(np.mean(x)) / sd) if sd > 0 else float("nan")
                )
        if "turnover" in df.columns:
            row["mean_turnover"] = float(np.nanmean(df["turnover"].to_numpy(dtype=float)))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    family_sheets: dict[str, pd.DataFrame] = {}
    for family in FAMILY_ORDER:
        rows = []
        for name, df in frames.items():
            if df is None or df.empty or strategy_family(name) != family:
                continue
            peer = df.copy()
            peer.insert(0, "strategy", str(name))
            peer = peer.rename(columns=rename)
            rows.append(peer)
        family_sheets[family] = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        reb.to_excel(writer, sheet_name="Rebalances", index=False)
        all_days.to_excel(writer, sheet_name="AllDays", index=False)
        trades.to_excel(writer, sheet_name="Trades", index=False)
        if not turnover.empty:
            turnover.to_excel(writer, sheet_name="Turnover", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        for family in FAMILY_ORDER:
            family_sheets[family].to_excel(writer, sheet_name=family, index=False)
    written["xlsx"] = str(out_xlsx)

    if out_csv is not None:
        out_csv = Path(out_csv)
        reb.to_csv(out_csv, index=False)
        written["csv"] = str(out_csv)
    return written
