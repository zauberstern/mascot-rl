"""Constituent allocation ledger for calendar IS / OOS forensic audits."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


class PortfolioAccountingLedger:
    """
    Records per-asset executed weights on real calendar dates.

    Use phase labels ``IS_HIST`` / ``OOS_TEST`` for historical PIT walks —
    do not invent calendar dates for synthetic rBergomi episodes.

    Optional ``strategy`` distinguishes HAPPO vs non-trivial baselines for
    academic side-by-side allocation tables.
    """

    def __init__(
        self,
        asset_names: list[str] | None = None,
        num_assets: int = 10,
    ):
        self.num_assets = int(num_assets)
        self.asset_names = (
            asset_names
            if asset_names and len(asset_names) == self.num_assets
            else [f"SECID_{i:03d}" for i in range(self.num_assets)]
        )
        self.records: list[dict] = []

    def record_step(
        self,
        date: str | pd.Timestamp,
        phase: str,
        weights_exec: torch.Tensor | np.ndarray,
        deltas: torch.Tensor | np.ndarray | None = None,
        weights_raw: torch.Tensor | np.ndarray | None = None,
        step_pnl: float = 0.0,
        step: int = 0,
        episode: int = 0,
        strategy: str = "happo",
    ) -> None:
        w_exec = _to_1d(weights_exec, self.num_assets)
        w_raw = (
            _to_1d(weights_raw, self.num_assets)
            if weights_raw is not None
            else np.full(self.num_assets, np.nan)
        )
        d_vec = (
            _to_1d(deltas, self.num_assets)
            if deltas is not None
            else np.zeros(self.num_assets)
        )
        ts = pd.Timestamp(date)
        strat = str(strategy or "happo")
        for k in range(self.num_assets):
            name = self.asset_names[k]
            self.records.append(
                {
                    "date": ts,
                    "phase": phase,
                    "strategy": strat,
                    "episode": int(episode),
                    "step": int(step),
                    "ticker": name,  # human-readable (AAPL, MSFT, …)
                    "asset_id": name,  # alias used by pivots/plots
                    "weight_raw": float(w_raw[k]),
                    "weight_exec": float(w_exec[k]),
                    "delta": float(d_vec[k]),
                    "step_pnl": float(step_pnl),
                }
            )

    def to_dataframe(self) -> pd.DataFrame:
        cols = [
            "date",
            "phase",
            "strategy",
            "episode",
            "step",
            "ticker",
            "asset_id",
            "weight_raw",
            "weight_exec",
            "delta",
            "step_pnl",
        ]
        if not self.records:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(self.records)

    def export_excel(self, output_path: Path) -> Path | None:
        df = self.to_dataframe()
        if df.empty:
            return None
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        group_keys = ["phase", "strategy", "date"]
        pivot_exec = (
            df.pivot_table(
                index=group_keys,
                columns="asset_id",
                values="weight_exec",
                aggfunc="last",
            )
            .fillna(0.0)
            .sort_index()
        )
        pivot_raw = (
            df.pivot_table(
                index=group_keys,
                columns="asset_id",
                values="weight_raw",
                aggfunc="last",
            )
            .fillna(0.0)
            .sort_index()
        )
        summary = (
            df.groupby(["phase", "strategy", "ticker"])
            .agg(
                weight_exec_mean=("weight_exec", "mean"),
                weight_exec_std=("weight_exec", "std"),
                weight_exec_min=("weight_exec", "min"),
                weight_exec_max=("weight_exec", "max"),
                step_pnl_mean=("step_pnl", "mean"),
            )
            .reset_index()
        )
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="Constituent Summary", index=False)
            pivot_exec.to_excel(writer, sheet_name="Executed Weights")
            pivot_raw.to_excel(writer, sheet_name="Raw Weights")
            # Cap stream size for Excel comfort.
            df.head(200_000).to_excel(writer, sheet_name="Transaction Stream", index=False)
            # Per-strategy summary sheet for baseline comparison.
            strategies = sorted(df["strategy"].dropna().unique().tolist())
            if len(strategies) > 1:
                for strat in strategies:
                    sub = summary[summary["strategy"] == strat]
                    sheet = str(strat)[:28] or "strategy"
                    sub.to_excel(writer, sheet_name=sheet, index=False)
        return output_path

    def export_parquet(self, output_path: Path) -> Path | None:
        df = self.to_dataframe()
        if df.empty:
            return None
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        return output_path


def _to_1d(x: torch.Tensor | np.ndarray, n: int) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().float().cpu().numpy()
    else:
        arr = np.asarray(x, dtype=np.float64)
    arr = np.ravel(arr)
    if arr.size < n:
        out = np.zeros(n, dtype=np.float64)
        out[: arr.size] = arr
        return out
    return arr[:n].astype(np.float64)
