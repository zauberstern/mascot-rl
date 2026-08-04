"""Trailing-Sharpe hold-leader and daily rolling leader (causal one-hot)."""
from __future__ import annotations

import numpy as np

from src.eval.onehot_leader import onehot, pick_leader, trailing_sharpe


def hold_leader(
    returns: np.ndarray,
    *,
    lookback: int,
    hold: int,
    min_obs: int = 20,
) -> np.ndarray:
    """W[t] one-hot. Uniform on [0, hold). Review at t=hold, 2*hold, ... using R[:t]."""
    R = np.asarray(returns, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError("returns must be (T, n)")
    t_len, n = R.shape
    if n < 2:
        raise ValueError("need at least 2 experts")
    h = int(hold)
    lb = int(lookback)
    if h < 1 or lb < 1:
        raise ValueError("lookback and hold must be >= 1")
    W = np.zeros((t_len, n), dtype=np.float64)
    uniform = np.ones(n, dtype=np.float64) / n
    if t_len == 0:
        return W
    end_warm = min(h, t_len)
    W[:end_warm] = uniform
    incumbent: int | None = None
    t = h
    while t < t_len:
        scores = np.array(
            [
                trailing_sharpe(R[:, i], max(0, t - lb), t, min_obs=min_obs)
                for i in range(n)
            ],
            dtype=np.float64,
        )
        lead = pick_leader(scores, incumbent=incumbent)
        w = uniform if lead is None else onehot(n, lead)
        if lead is not None:
            incumbent = lead
        end = min(t_len, t + h)
        W[t:end] = w
        t = end
    return W


def rolling_leader(
    returns: np.ndarray,
    *,
    lookback: int,
    min_obs: int = 20,
) -> np.ndarray:
    """Each t: if t < lookback uniform; else pick_leader on [t-lookback, t)."""
    R = np.asarray(returns, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError("returns must be (T, n)")
    t_len, n = R.shape
    if n < 2:
        raise ValueError("need at least 2 experts")
    lb = int(lookback)
    W = np.zeros((t_len, n), dtype=np.float64)
    uniform = np.ones(n, dtype=np.float64) / n
    incumbent: int | None = None
    for t in range(t_len):
        if t < lb:
            W[t] = uniform
            continue
        scores = np.array(
            [
                trailing_sharpe(R[:, i], t - lb, t, min_obs=min_obs)
                for i in range(n)
            ],
            dtype=np.float64,
        )
        lead = pick_leader(scores, incumbent=incumbent)
        if lead is None:
            W[t] = uniform
            incumbent = None
        else:
            W[t] = onehot(n, lead)
            incumbent = lead
    return W
