"""Tests for performance-sleeping Variable-Share."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.performance_sleeping import performance_sleeping


def test_performance_sleeping_downweights_chronic_loser() -> None:
    t, n = 200, 3
    R = np.full((t, n), 0.001, dtype=np.float64)
    R[:, 0] = -0.001
    W = performance_sleeping(R, alpha=0.05, lookback=50, min_obs=20, owl_index=2)
    assert W.shape == (t, n)
    np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-8)
    # After lookback, expert 0 should have less mass than equal
    assert float(W[150, 0]) < 1.0 / n


def test_performance_sleeping_pit() -> None:
    rng = np.random.default_rng(9)
    R = rng.normal(0.0, 0.01, size=(100, 3))
    W0 = performance_sleeping(R, alpha=0.03, lookback=40)
    day = 55
    R2 = R.copy()
    R2[day] += 0.05
    W1 = performance_sleeping(R2, alpha=0.03, lookback=40)
    np.testing.assert_allclose(W0[day], W1[day], rtol=1e-12)


def test_performance_sleeping_prefix() -> None:
    rng = np.random.default_rng(10)
    R = rng.normal(0.0005, 0.01, size=(120, 3))
    W = performance_sleeping(R, alpha=0.02, lookback=30)
    t = 80
    Wp = performance_sleeping(R[:t], alpha=0.02, lookback=30)
    np.testing.assert_allclose(W[:t], Wp, rtol=1e-10, atol=1e-12)


def test_performance_sleeping_no_turbulence_import() -> None:
    import ast

    src = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "eval"
        / "performance_sleeping.py"
    )
    tree = ast.parse(src.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "turbulence" not in alias.name
        if isinstance(node, ast.ImportFrom):
            assert "turbulence" not in (node.module or "")
