"""Shuffled-label falsification: Sharpe must collapse under within-date permute."""
from __future__ import annotations

import numpy as np

from src.eval.stats_rigor import annualized_sharpe


def test_shuffled_labels_collapse_sharpe() -> None:
    rng = np.random.default_rng(0)
    t, k = 252, 20
    # Plant a weak cross-sectional signal in returns
    signal = rng.normal(size=(t, k))
    returns = 0.001 * signal + 0.01 * rng.normal(size=(t, k))
    # Long-short on contemporaneous signal (cheating) → positive Sharpe
    w = signal - signal.mean(axis=1, keepdims=True)
    w = w / (np.abs(w).sum(axis=1, keepdims=True) + 1e-12)
    pnl = (w * returns).sum(axis=1)
    sr_cheat = annualized_sharpe(pnl)
    # Shuffle labels within each date → destroy CS signal
    shuffled = returns.copy()
    for i in range(t):
        shuffled[i] = rng.permutation(shuffled[i])
    pnl_s = (w * shuffled).sum(axis=1)
    sr_shuf = annualized_sharpe(pnl_s)
    assert sr_cheat > 0.5
    assert abs(sr_shuf) < abs(sr_cheat) * 0.5
