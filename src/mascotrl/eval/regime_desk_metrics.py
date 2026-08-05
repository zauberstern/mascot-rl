"""Desk-level metrics for Ch.10 Fixed-Share regime desk (eval only)."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def sharpe_annualized(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    sd = float(np.std(r, ddof=1))
    if sd <= 1e-15:
        return 0.0
    return float(np.mean(r) / sd * math.sqrt(252.0))


def max_drawdown(cumwealth: np.ndarray) -> float:
    """Max drawdown from a cumulative wealth path (peak-to-trough)."""
    w = np.asarray(cumwealth, dtype=np.float64).reshape(-1)
    if w.size == 0 or not np.isfinite(w).any():
        return float("nan")
    w = np.where(np.isfinite(w), w, np.nan)
    peak = np.maximum.accumulate(np.nan_to_num(w, nan=0.0))
    # Prefer true peak tracking with NaN skip
    peak = np.empty_like(w)
    running = np.nan
    for i, v in enumerate(w):
        if np.isfinite(v):
            running = v if not np.isfinite(running) else max(running, v)
        peak[i] = running
    dd = (w - peak) / np.where(np.abs(peak) > 1e-15, peak, np.nan)
    m = np.nanmin(dd)
    return float(m) if np.isfinite(m) else float("nan")


def weight_turnover_l1(weights: np.ndarray) -> float:
    """Mean half L1 weight change: 0.5 * mean_t ||W_t - W_{t-1}||_1."""
    W = np.asarray(weights, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] < 2:
        return float("nan")
    diff = np.abs(W[1:] - W[:-1]).sum(axis=1)
    return float(0.5 * np.mean(diff))


def per_regime_desk_stats(
    returns_by_book: Mapping[str, np.ndarray],
    turbulent: np.ndarray,
) -> dict[str, Any]:
    """Sharpe / mean / n_days on turbulent vs calm masks."""
    mask = np.asarray(turbulent, dtype=bool).reshape(-1)
    out: dict[str, Any] = {"turbulent": {}, "calm": {}}
    for regime_name, m in (("turbulent", mask), ("calm", ~mask)):
        bucket: dict[str, Any] = {}
        for book, rets in returns_by_book.items():
            r = np.asarray(rets, dtype=np.float64).reshape(-1)
            if r.shape[0] != mask.shape[0]:
                bucket[book] = {
                    "sharpe": float("nan"),
                    "mean": float("nan"),
                    "n_days": 0,
                }
                continue
            sub = r[m]
            sub = sub[np.isfinite(sub)]
            bucket[book] = {
                "sharpe": sharpe_annualized(sub),
                "mean": float(np.mean(sub)) if sub.size else float("nan"),
                "n_days": int(sub.size),
            }
        out[regime_name] = bucket
    return out


def best_solo_expert(
    losses: np.ndarray,
    returns: np.ndarray,
    names: Sequence[str],
) -> dict[str, Any]:
    """Hindsight single expert = argmin cumulative loss."""
    L = np.asarray(losses, dtype=np.float64)
    R = np.asarray(returns, dtype=np.float64)
    if L.ndim != 2 or R.shape != L.shape or len(names) != L.shape[1]:
        raise ValueError("losses/returns/names shape mismatch")
    cum = L.sum(axis=0)
    idx = int(np.argmin(cum))
    r = R[:, idx]
    wealth = np.cumprod(1.0 + np.nan_to_num(r, nan=0.0))
    return {
        "name": str(names[idx]),
        "index": idx,
        "total_loss": float(cum[idx]),
        "sharpe": sharpe_annualized(r),
        "max_drawdown": max_drawdown(wealth),
    }


def book_table_row(
    returns: np.ndarray,
    *,
    turnover: float | None = None,
) -> dict[str, Any]:
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    wealth = np.cumprod(1.0 + np.nan_to_num(r, nan=0.0))
    return {
        "sharpe": sharpe_annualized(r),
        "max_drawdown": max_drawdown(wealth),
        "turnover": float(turnover) if turnover is not None else float("nan"),
    }
