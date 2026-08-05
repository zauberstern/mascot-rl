"""Tests for Herbster-Warmuth Variable-Share."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mascotrl.eval.fixed_share import fixed_share
from mascotrl.eval.variable_share import variable_share


def _four_regime_losses_01(t_per: int = 200, n: int = 4) -> np.ndarray:
    """Unit-interval block losses: expert i is best (0) in regime i; others 1."""
    losses = np.ones((t_per * n, n), dtype=np.float64)
    for i in range(n):
        start = i * t_per
        end = start + t_per
        losses[start:end, i] = 0.0
    return losses


def test_variable_share_weights_sum_and_track_blocks() -> None:
    losses = _four_regime_losses_01(200, 4)
    alpha = 3 / 799
    w = variable_share(losses, alpha=alpha, eta=0.5)
    assert w.shape == losses.shape
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-10)
    for i in range(4):
        mid = i * 200 + 100
        assert int(np.argmax(w[mid])) == i


def test_variable_share_less_sharing_than_fs_on_perfect_expert() -> None:
    t, n = 200, 3
    L = np.ones((t, n), dtype=np.float64)
    L[:, 0] = 0.0
    alpha = 0.05
    w_vs = variable_share(L, alpha=alpha, eta=0.5)
    w_fs = fixed_share(L, alpha=alpha, eta=0.5)
    mid = t // 2
    assert w_vs[mid, 0] > w_fs[mid, 0]


def test_variable_share_prefix_stable() -> None:
    losses = _four_regime_losses_01(40, 3)
    w_full = variable_share(losses, alpha=0.02, eta=0.5)
    t = 50
    w_pref = variable_share(losses[:t], alpha=0.02, eta=0.5)
    np.testing.assert_allclose(w_full[:t], w_pref, rtol=1e-10, atol=1e-12)


def test_variable_share_rejects_out_of_range() -> None:
    L = np.array([[0.0, 1.5]], dtype=np.float64)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        variable_share(L, alpha=0.1, eta=0.5)


def test_variable_share_no_turbulence_import() -> None:
    import ast

    src = Path(__file__).resolve().parents[1] / "src" / "mascotrl" / "eval" / "variable_share.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "turbulence" not in alias.name
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert "turbulence" not in mod
            for alias in node.names:
                assert alias.name != "turbulence_index"
