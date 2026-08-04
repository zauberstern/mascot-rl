"""Herbster-Warmuth Fixed-Share: forward-only expert tracking."""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.fixed_share import fixed_share, pre_register_alpha


def _four_regime_losses(t_per: int = 200, n: int = 4) -> np.ndarray:
    """Synthetic losses: expert i is best in regime i (low loss)."""
    losses = np.ones((t_per * n, n), dtype=np.float64)
    for i in range(n):
        start = i * t_per
        end = start + t_per
        losses[start:end, i] = 0.0
        # Others mediocre.
        losses[start:end, :] += 0.1
        losses[start:end, i] = 0.0
    return losses


def test_fixed_share_weights_sum_to_one() -> None:
    losses = _four_regime_losses(50, 3)
    w = fixed_share(losses, alpha=0.05, eta=0.5)
    assert w.shape == losses.shape
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(w >= -1e-12)


def test_fixed_share_tracks_synthetic_regime_switches() -> None:
    losses = _four_regime_losses(200, 4)
    # k=3 switches expected over T=800 -> alpha* = 3/(800-1)
    alpha = pre_register_alpha(k_switches=3, sequence_length=800)
    w = fixed_share(losses, alpha=alpha, eta=0.5)
    # In the middle of each regime block, dominant weight should be that expert.
    for i in range(4):
        mid = i * 200 + 100
        assert int(np.argmax(w[mid])) == i


def test_fixed_share_no_lookahead_prefix_stability() -> None:
    losses = _four_regime_losses(80, 3)
    w_full = fixed_share(losses, alpha=0.02, eta=0.5)
    t = 100
    w_pref = fixed_share(losses[:t], alpha=0.02, eta=0.5)
    np.testing.assert_allclose(w_full[:t], w_pref, rtol=1e-10, atol=1e-12)


def test_pre_register_alpha_formula() -> None:
    assert pre_register_alpha(k_switches=3, sequence_length=801) == pytest.approx(3 / 800)
    with pytest.raises(ValueError):
        pre_register_alpha(k_switches=1, sequence_length=1)


def test_fixed_share_no_turbulence_index_import() -> None:
    """Fixed-Share module must not import turbulence (α is Herbster prior only)."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "eval" / "fixed_share.py"
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
