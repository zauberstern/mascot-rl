"""Causal / cross-sectional normalization invariants for feature blocks."""
from __future__ import annotations

import numpy as np

from src.features.blocks.normalize import (
    cross_sectional_zscore,
    expanding_causal_zscore,
    winsorize_panel,
)


def test_winsorize_1_99():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 5))
    x[0, 0] = 100.0
    x[1, 0] = -100.0
    out = winsorize_panel(x, lower_q=0.01, upper_q=0.99)
    assert out.max() < 50.0
    assert out.min() > -50.0


def test_cross_sectional_zscore_mean_near_zero():
    rng = np.random.default_rng(1)
    # (T, K) panel — z across names per date.
    panel = rng.normal(loc=2.0, scale=3.0, size=(30, 20))
    z = cross_sectional_zscore(panel, clip=3.0)
    # Per-date cross-sectional mean ~0 (finite names).
    means = np.nanmean(z, axis=1)
    assert np.nanmax(np.abs(means)) < 1e-10
    assert np.nanmax(np.abs(z)) <= 3.0 + 1e-9


def test_expanding_causal_zscore_no_lookahead():
    rng = np.random.default_rng(2)
    # Time-series panel (T, K); expanding z uses only past+present.
    x = rng.normal(size=(40, 4))
    t = 20
    z0 = expanding_causal_zscore(x)
    mutated = x.copy()
    mutated[t + 1 :] = 999.0
    z1 = expanding_causal_zscore(mutated)
    np.testing.assert_allclose(z0[: t + 1], z1[: t + 1])
    # Past mutation must change current z.
    mutated2 = x.copy()
    mutated2[t - 5, 0] += 1.0
    z2 = expanding_causal_zscore(mutated2)
    assert not np.allclose(z0[t, 0], z2[t, 0])
