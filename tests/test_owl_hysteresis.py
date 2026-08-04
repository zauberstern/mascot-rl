"""Tests for Owl hysteresis switcher."""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.owl_hysteresis import owl_hysteresis


def test_owl_hysteresis_requires_owl() -> None:
    R = np.zeros((50, 2))
    with pytest.raises(ValueError, match="owl"):
        owl_hysteresis(R, ["fox", "tortoise"], lookback=20, margin=0.25)


def test_owl_hysteresis_switches_to_specialist_then_back() -> None:
    t, n = 400, 3
    names = ["fox", "cheetah", "owl"]
    rng = np.random.default_rng(11)
    R = rng.normal(0.0003, 0.002, size=(t, n))
    # fox dominates days 150-280 strongly
    R[150:280, 0] += 0.01
    lookback = 60
    W = owl_hysteresis(R, names, lookback=lookback, margin=0.25, min_obs=20)
    np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-10)
    # mid specialist block: after window fills with fox edge
    mid = 220
    assert int(np.argmax(W[mid])) == 0
    # late: back toward owl after fox edge leaves window
    late = 360
    assert int(np.argmax(W[late])) == 2


def test_owl_hysteresis_prefix_and_pit() -> None:
    names = ["fox", "owl"]
    rng = np.random.default_rng(12)
    R = rng.normal(0.0002, 0.005, size=(200, 2))
    W = owl_hysteresis(R, names, lookback=40, margin=0.25)
    t = 120
    Wp = owl_hysteresis(R[:t], names, lookback=40, margin=0.25)
    np.testing.assert_allclose(W[:t], Wp, rtol=1e-10, atol=1e-12)
    day = 80
    R2 = R.copy()
    R2[day] += 0.1
    W2 = owl_hysteresis(R2, names, lookback=40, margin=0.25)
    np.testing.assert_allclose(W[day], W2[day], rtol=1e-12)
