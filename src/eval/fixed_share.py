"""Herbster-Warmuth Fixed-Share expert tracking (forward-only).

Primary Ch.10 real-time rotator across bot archetypes. Alpha must be
pre-registered from a switch-frequency prior, never tuned on OOS.
"""
from __future__ import annotations

import numpy as np


def pre_register_alpha(*, k_switches: int, sequence_length: int) -> float:
    """Herbster-Warmuth α* = k / (ℓ - 1) from a locked switch prior."""
    k = int(k_switches)
    ell = int(sequence_length)
    if k < 0:
        raise ValueError("k_switches must be >= 0")
    if ell < 2:
        raise ValueError("sequence_length must be >= 2")
    return float(k) / float(ell - 1)


def share_update(w_m: np.ndarray, alpha: float) -> np.ndarray:
    """One Fixed-Share pool step on post-loss weights; renormalize to simplex."""
    wm = np.asarray(w_m, dtype=np.float64).reshape(-1)
    n = int(wm.size)
    if n < 2:
        raise ValueError("need at least 2 experts")
    a = float(alpha)
    if not (0.0 <= a <= 1.0):
        raise ValueError(f"alpha must be in [0,1]; got {alpha}")
    pool = a * float(wm.sum())
    w = (1.0 - a) * wm + (pool - a * wm) / (n - 1)
    s = float(w.sum())
    if s <= 0.0 or not np.isfinite(s):
        return np.ones(n, dtype=np.float64) / n
    return w / s


def fixed_share(
    losses: np.ndarray,
    *,
    alpha: float,
    eta: float,
) -> np.ndarray:
    """Return weight matrix W[t, i] used at trial t (before observing loss t).

    Loss update then Fixed-Share pool redistribution, renormalized each step.
    """
    L = np.asarray(losses, dtype=np.float64)
    if L.ndim != 2:
        raise ValueError("losses must be (T, n)")
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
        wm = w * np.exp(-e * L[t])
        w = share_update(wm, a)
    return hist
