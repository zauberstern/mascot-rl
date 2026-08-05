"""Tests for Page-Hinkley one-hot switcher."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mascotrl.eval.page_hinkley_switch import page_hinkley_switch


def test_page_hinkley_switches_after_crash() -> None:
    t, n = 200, 3
    names = ["a", "b", "c"]  # no owl => start on index 0
    R = np.full((t, n), 0.001, dtype=np.float64)
    R[:100, 0] = 0.01
    R[100:, 0] = -0.02
    R[:, 1] = 0.002
    R[:, 2] = 0.0015
    W = page_hinkley_switch(R, names, delta=1e-4, lam=0.02)
    assert W.shape == R.shape
    np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-10)
    assert int(np.argmax(W[150])) != 0


def test_page_hinkley_pit_R_t() -> None:
    names = ["a", "b", "c"]
    rng = np.random.default_rng(5)
    R = rng.normal(0.001, 0.01, size=(120, 3))
    W0 = page_hinkley_switch(R, names, delta=1e-4, lam=0.02)
    day = 60
    R2 = R.copy()
    R2[day] += 0.05
    W1 = page_hinkley_switch(R2, names, delta=1e-4, lam=0.02)
    np.testing.assert_allclose(W0[day], W1[day], rtol=1e-12)


def test_page_hinkley_prefix_stable() -> None:
    names = ["a", "b", "owl"]
    rng = np.random.default_rng(6)
    R = rng.normal(0.001, 0.01, size=(100, 3))
    W = page_hinkley_switch(R, names)
    t = 70
    Wp = page_hinkley_switch(R[:t], names)
    np.testing.assert_allclose(W[:t], Wp, rtol=1e-10, atol=1e-12)


def test_page_hinkley_no_turbulence_import() -> None:
    import ast

    src = Path(__file__).resolve().parents[1] / "src" / "mascotrl" / "eval" / "page_hinkley_switch.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "turbulence" not in alias.name
        if isinstance(node, ast.ImportFrom):
            assert "turbulence" not in (node.module or "")
