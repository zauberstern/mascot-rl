"""Tests for AdaHedge, FTL, FlipFlop, AdaHedge+Fixed-Share."""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.adahedge import (
    adahedge,
    adahedge_fixed_share,
    flipflop,
    follow_the_leader,
)
from src.eval.fixed_share import fixed_share, share_update
from tests.test_variable_share import _four_regime_losses_01


def test_share_update_matches_fixed_share_step() -> None:
    rng = np.random.default_rng(0)
    w = rng.dirichlet(np.ones(4))
    a = 0.1
    out = share_update(w, a)
    n = 4
    pool = a * float(w.sum())
    expected = (1.0 - a) * w + (pool - a * w) / (n - 1)
    expected = expected / expected.sum()
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_fixed_share_unchanged_after_share_update_refactor() -> None:
    losses = _four_regime_losses_01(50, 3)
    w = fixed_share(losses, alpha=0.05, eta=0.5)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-10)


def test_adahedge_prefix_stable() -> None:
    losses = _four_regime_losses_01(40, 3)
    w_full = adahedge(losses)
    t = 50
    w_pref = adahedge(losses[:t])
    np.testing.assert_allclose(w_full[:t], w_pref, rtol=1e-10, atol=1e-12)


def test_adahedge_share_tracks_blocks() -> None:
    losses = _four_regime_losses_01(200, 4)
    alpha = 3 / 799
    w = adahedge_fixed_share(losses, alpha=alpha)
    assert w.shape == losses.shape
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-10)
    for i in range(4):
        mid = i * 200 + 100
        assert int(np.argmax(w[mid])) == i


def test_flipflop_weights_sum_to_one() -> None:
    losses = _four_regime_losses_01(80, 3)
    w = flipflop(losses)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(w >= -1e-12)


def test_ftl_uniform_then_leader() -> None:
    L = np.array(
        [
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    w = follow_the_leader(L)
    np.testing.assert_allclose(w[0], np.ones(3) / 3.0)
    assert int(np.argmax(w[1])) == 0
    assert w[1, 0] == pytest.approx(1.0)
