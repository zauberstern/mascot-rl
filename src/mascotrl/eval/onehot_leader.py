"""Shared one-hot leader helpers for causal desk switchers."""
from __future__ import annotations

import math

import numpy as np

from src.eval.regime_desk_metrics import sharpe_annualized


def trailing_sharpe(
    col: np.ndarray,
    start: int,
    end: int,
    *,
    min_obs: int = 20,
) -> float:
    """sharpe_annualized on col[start:end] (end exclusive). NaN if < min_obs finite."""
    x = np.asarray(col, dtype=np.float64).reshape(-1)[int(start) : int(end)]
    finite = x[np.isfinite(x)]
    if finite.size < int(min_obs):
        return float("nan")
    return float(sharpe_annualized(finite))


def onehot(n: int, i: int) -> np.ndarray:
    w = np.zeros(int(n), dtype=np.float64)
    w[int(i)] = 1.0
    return w


def pick_leader(scores: np.ndarray, *, incumbent: int | None) -> int | None:
    """Among finite scores, argmax. Ties: incumbent if in argmax set, else lowest index."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = np.isfinite(s)
    if not finite.any():
        return None
    m = float(np.nanmax(s))
    cands = [i for i in range(s.size) if finite[i] and abs(float(s[i]) - m) <= 1e-15]
    if not cands:
        return None
    if incumbent is not None and int(incumbent) in cands:
        return int(incumbent)
    return int(min(cands))
