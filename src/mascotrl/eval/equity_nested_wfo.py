"""Equity nested walk-forward alongside CPCV (W6).

Expanding-window fine-tune on ``HistoricalArmEnv``. Explicitly stamped
``is_cpcv: false`` — WFO is never a capital-gate substitute for CPCV.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from mascotrl.eval.friction import friction_spec_from_cfg
from mascotrl.eval.research_alpha_train import train_research_hist
from mascotrl.eval.stats_rigor import annualized_sharpe


def _expanding_folds(
    n: int,
    *,
    n_folds: int = 5,
    min_train: int = 60,
    test_frac: float = 0.15,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build expanding train / forward test index pairs over ``0..n-1``."""
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    test_len = max(10, int(round(n * float(test_frac) / max(n_folds, 1))))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    # Last fold ends at n; walk backward so each test block is contiguous.
    cursor = n
    for _ in range(int(n_folds)):
        test_end = cursor
        test_start = max(min_train, test_end - test_len)
        train_end = test_start
        if train_end < min_train:
            break
        train_idx = np.arange(0, train_end, dtype=int)
        test_idx = np.arange(test_start, test_end, dtype=int)
        if train_idx.size < min_train or test_idx.size < 5:
            break
        folds.append((train_idx, test_idx))
        cursor = test_start
    folds.reverse()
    return folds


def run_equity_nested_wfo(
    dates: Sequence[pd.Timestamp],
    returns: np.ndarray,
    factors: np.ndarray,
    cfg: Mapping[str, Any],
    *,
    seed: int = 0,
    n_folds: int = 5,
) -> dict[str, Any]:
    """Train on expanding windows; score each forward block after costs.

    Returns a report stamped ``protocol: expanding_window_wfo`` and
    ``is_cpcv: false``. Does not emit capital-allocation claim fields.
    """
    dates = list(dates)
    rets = np.asarray(returns, dtype=np.float64)
    fac = np.asarray(factors, dtype=np.float64)
    if len(dates) != rets.shape[0]:
        raise ValueError("dates length must match returns T")
    folds = _expanding_folds(rets.shape[0], n_folds=int(n_folds))
    fold_reports: list[dict[str, Any]] = []
    oos_pnls: list[float] = []

    # Import locally to avoid circular imports at module load.
    from mascotrl.eval.research_alpha_cpcv import _roll_test_pnl, _slice_feature_extras
    from mascotrl.eval.residualization import fit_ff4_residualizer, freeze_residualizer
    from mascotrl.eval.yaml_honesty import track_copy

    cfg_local = track_copy(cfg)
    fric = friction_spec_from_cfg(cfg_local)

    for i, (train_idx, test_idx) in enumerate(folds):
        train_out = train_research_hist(
            rets[train_idx],
            fac[train_idx],
            _slice_feature_extras(cfg_local, train_idx),
            seed=int(seed) + i,
            agent=None,
        )
        agent = train_out["agent"]
        train_resid = freeze_residualizer(
            fit_ff4_residualizer(
                np.nanmean(rets[train_idx], axis=1),
                fac[train_idx],
                fold_id=f"nested_wfo_fold_{i}",
            ),
            f"nested_wfo_fold_{i}",
        )
        pnl_by_date = _roll_test_pnl(
            returns=rets,
            factors=fac,
            dates=dates,
            idx=test_idx,
            agent=agent,
            cfg=_slice_feature_extras(cfg_local, test_idx),
            friction=fric,
            train_residualizer=train_resid,
        )
        series = [
            float(v.get("total_net", float("nan")))
            for _, v in sorted(pnl_by_date.items(), key=lambda kv: kv[0])
            if isinstance(v, dict)
        ]
        sh = (
            annualized_sharpe(
                np.asarray(series, dtype=float),
                periods=float(cfg_local.get("_periods_per_year") or 252.0),
            )
            if series
            else float("nan")
        )
        fold_reports.append(
            {
                "fold_id": i,
                "n_train": int(train_idx.size),
                "n_test": int(test_idx.size),
                "sharpe": float(sh) if np.isfinite(sh) else None,
                "train_start": str(dates[int(train_idx[0])].date()),
                "train_end": str(dates[int(train_idx[-1])].date()),
                "test_start": str(dates[int(test_idx[0])].date()),
                "test_end": str(dates[int(test_idx[-1])].date()),
            }
        )
        oos_pnls.extend(series)

    sharpes = [
        float(f["sharpe"])
        for f in fold_reports
        if f.get("sharpe") is not None and np.isfinite(float(f["sharpe"]))
    ]
    return {
        "protocol": "expanding_window_wfo",
        "is_cpcv": False,
        "n_folds": len(fold_reports),
        "folds": fold_reports,
        "sharpe_mean": float(np.mean(sharpes)) if sharpes else float("nan"),
        "sharpe_std": float(np.std(sharpes)) if len(sharpes) > 1 else 0.0,
        "positive_fold_rate": (
            float(np.mean([s > 0.0 for s in sharpes])) if sharpes else float("nan")
        ),
        "oos_sharpe": annualized_sharpe(
            np.asarray(oos_pnls, dtype=float),
            periods=float(cfg_local.get("_periods_per_year") or 252.0),
        )
        if oos_pnls
        else float("nan"),
        "note": (
            "Expanding-window WFO is a fine-tune diagnostic reported alongside "
            "CPCV. It is not CPCV and cannot satisfy capital gates."
        ),
    }
