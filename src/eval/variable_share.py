"""Herbster-Warmuth Variable-Share (losses in [0, 1]).

A currently-perfect expert (L=0) does not share mass. Requires unit-interval
losses (square / hellinger / absolute after causal scaling).
"""
from __future__ import annotations

import numpy as np


def variable_share(
    losses: np.ndarray,
    *,
    alpha: float,
    eta: float = 0.5,
) -> np.ndarray:
    """Return W[t, i] used at trial t (before observing loss t).

    Herbster Fig. 1 Variable-share update. Losses must lie in [0, 1].
    """
    L = np.asarray(losses, dtype=np.float64)
    if L.ndim != 2:
        raise ValueError("losses must be (T, n)")
    if np.any(L < -1e-12) or np.any(L > 1.0 + 1e-12):
        raise ValueError("Variable-Share requires losses in [0,1]")
    L = np.clip(L, 0.0, 1.0)
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in [0,1]; got {alpha}")
    if float(eta) <= 0.0:
        raise ValueError(f"eta must be > 0; got {eta}")
    t_len, n = L.shape
    if n < 2:
        raise ValueError("need at least 2 experts")
    w = np.ones(n, dtype=np.float64) / n
    hist = np.zeros((t_len, n), dtype=np.float64)
    a = float(alpha)
    e = float(eta)
    for t in range(t_len):
        hist[t] = w
        lt = L[t]
        wm = w * np.exp(-e * lt)
        # share fraction 1 - (1-alpha)^L_i
        one_m_a = 1.0 - a
        share_frac = 1.0 - np.power(one_m_a, lt)
        keep = np.power(one_m_a, lt)
        pool = float(np.dot(share_frac, wm))
        w = keep * wm + (pool - share_frac * wm) / (n - 1)
        s = float(w.sum())
        if s <= 0.0 or not np.isfinite(s):
            w = np.ones(n, dtype=np.float64) / n
        else:
            w = w / s
    return hist
