"""Helmbold EG on experts-as-assets and Wintenberger BOA on [0,1] losses."""
from __future__ import annotations

import numpy as np

from mascotrl.eval.fixed_share import share_update


def eg_experts(returns: np.ndarray, *, eta: float = 0.05) -> np.ndarray:
    """Helmbold EG. W[0]=1/n. Emit W[t], observe R[t], update on price relatives."""
    R = np.asarray(returns, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError("returns must be (T, n)")
    if float(eta) <= 0.0:
        raise ValueError(f"eta must be > 0; got {eta}")
    t_len, n = R.shape
    if n < 2:
        raise ValueError("need at least 2 experts")
    w = np.ones(n, dtype=np.float64) / n
    hist = np.zeros((t_len, n), dtype=np.float64)
    e = float(eta)
    for t in range(t_len):
        hist[t] = w
        x = 1.0 + R[t]
        port = float(np.dot(w, x))
        if port < 1e-12:
            port = 1e-12
        w = w * np.exp(e * x / port)
        s = float(w.sum())
        if s <= 0.0 or not np.isfinite(s):
            w = np.ones(n, dtype=np.float64) / n
        else:
            w = w / s
    return hist


def boa_experts(
    losses_01: np.ndarray,
    *,
    eta: float = 1.0,
    alpha: float | None = None,
) -> np.ndarray:
    """Wintenberger BOA on [0,1] losses; optional Fixed-Share after update."""
    L = np.asarray(losses_01, dtype=np.float64)
    if L.ndim != 2:
        raise ValueError("losses must be (T, n)")
    if np.any(L < -1e-12) or np.any(L > 1.0 + 1e-12):
        raise ValueError("BOA requires losses in [0,1]")
    L = np.clip(L, 0.0, 1.0)
    if float(eta) <= 0.0:
        raise ValueError(f"eta must be > 0; got {eta}")
    t_len, n = L.shape
    if n < 2:
        raise ValueError("need at least 2 experts")
    w = np.ones(n, dtype=np.float64) / n
    hist = np.zeros((t_len, n), dtype=np.float64)
    e = float(eta)
    for t in range(t_len):
        hist[t] = w
        ell = L[t]
        # w[i] *= exp(-eta * ell * (1 + eta * ell))
        wm = w * np.exp(-e * ell * (1.0 + e * ell))
        s = float(wm.sum())
        if s <= 0.0 or not np.isfinite(s):
            wm = np.ones(n, dtype=np.float64) / n
        else:
            wm = wm / s
        if alpha is not None:
            w = share_update(wm, float(alpha))
        else:
            w = wm
    return hist
