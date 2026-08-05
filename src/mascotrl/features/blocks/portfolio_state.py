"""Per-asset portfolio / implementation-state features (E5)."""
from __future__ import annotations

import numpy as np


def build_portfolio_state_features(
    w_prev: np.ndarray,
    days_held: np.ndarray,
    cum_cost: np.ndarray,
) -> np.ndarray:
    """Stack implementation-state channels → ``(K, 3)`` or broadcast panel.

    Accepts:
      - 1-D ``(K,)`` → returns ``(K, 3)``
      - 2-D ``(T, K)`` → returns ``(T, K, 3)``
    """
    w = np.asarray(w_prev, dtype=np.float64)
    d = np.asarray(days_held, dtype=np.float64)
    c = np.asarray(cum_cost, dtype=np.float64)
    if w.shape != d.shape or w.shape != c.shape:
        raise ValueError(
            f"w_prev/days_held/cum_cost shape mismatch: {w.shape}, {d.shape}, {c.shape}"
        )
    if w.ndim == 1:
        return np.stack([w, d, c], axis=-1)
    if w.ndim == 2:
        return np.stack([w, d, c], axis=-1)
    raise ValueError(f"expected 1-D or 2-D inputs, got ndim={w.ndim}")
