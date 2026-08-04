"""P6: Sharpe-ratio difference test (policy vs peer) via stationary bootstrap."""
from __future__ import annotations

import numpy as np

from src.eval.ledoit_wolf_sharpe import sharpe_difference_test


def test_identical_series_delta_near_zero_high_pvalue() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, size=400)
    out = sharpe_difference_test(r, r.copy(), n_boot=199, block_mean=5, seed=0)
    assert out["method"] == "stationary_bootstrap_delta"
    assert abs(out["delta"]) < 1e-12
    assert abs(out["sharpe_a"] - out["sharpe_b"]) < 1e-12
    assert out["pvalue"] > 0.5
    assert out["n_obs"] == 400
    assert out["ci_low"] <= out["delta"] <= out["ci_high"]


def test_clearly_better_series_positive_delta() -> None:
    rng = np.random.default_rng(1)
    n = 500
    b = rng.normal(0.0002, 0.01, size=n)
    # Policy has a large additive edge on the same noise.
    a = b + 0.003
    out = sharpe_difference_test(a, b, n_boot=299, block_mean=5, seed=1)
    assert out["delta"] > 0.0
    assert out["sharpe_a"] > out["sharpe_b"]
    assert out["n_obs"] == n
    assert np.isfinite(out["pvalue"])
    assert out["ci_low"] < out["ci_high"]


def test_unequal_length_truncated_to_overlap() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(0.001, 0.01, size=300)
    b = rng.normal(0.0005, 0.01, size=180)
    out = sharpe_difference_test(a, b, n_boot=99, block_mean=5, seed=2)
    assert out["n_obs"] == 180
    assert np.isfinite(out["sharpe_a"])
    assert np.isfinite(out["sharpe_b"])
    assert np.isfinite(out["delta"])
