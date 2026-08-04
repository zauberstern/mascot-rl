"""Causal expert loss maps for desk mixers (eval only).

Log-wealth loss is mixable for portfolio wealth. Expanding [0,1] affine scale
is for Variable-Share / BOA; uses only past min/max (no look-ahead).
"""
from __future__ import annotations

import numpy as np


def log_wealth_loss(returns: np.ndarray, *, floor: float = 1e-8) -> np.ndarray:
    """ell[t,i] = -log(max(1+R[t,i], floor))."""
    R = np.asarray(returns, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError("returns must be (T, n)")
    return -np.log(np.maximum(1.0 + R, float(floor)))


def expanding_unit_interval(ell: np.ndarray) -> np.ndarray:
    """Causal [0,1] scale. L[t] uses min/max of ell[:t] (past only).

    t=0: L[0]=0.5 (no past range). After observing ell[t] for the update,
    past is ell[:t] so W[t] already emitted is unaffected.
    """
    E = np.asarray(ell, dtype=np.float64)
    if E.ndim != 2:
        raise ValueError("ell must be (T, n)")
    t_len, n = E.shape
    out = np.empty_like(E)
    for t in range(t_len):
        if t == 0:
            out[t] = 0.5
            continue
        past = E[:t].reshape(-1)
        lo = float(np.min(past))
        hi = float(np.max(past))
        span = hi - lo
        if span <= 1e-15:
            out[t] = 0.5
        else:
            scaled = (E[t] - lo) / span
            out[t] = np.clip(scaled, 0.0, 1.0)
    return out
