"""Tests for one-hot leader helpers."""
from __future__ import annotations

import math

import numpy as np

from mascotrl.eval.onehot_leader import onehot, pick_leader, trailing_sharpe


def test_trailing_sharpe_constant_positive() -> None:
    rng = np.random.default_rng(1)
    r = 0.01 + rng.normal(0, 0.001, size=50)
    s = trailing_sharpe(r, 0, 50, min_obs=20)
    assert math.isfinite(s) and s > 0


def test_trailing_sharpe_min_obs_nan() -> None:
    r = np.full(10, 0.01, dtype=np.float64)
    assert math.isnan(trailing_sharpe(r, 0, 10, min_obs=20))


def test_trailing_sharpe_prefix_no_future() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, size=80)
    t = 40
    a = trailing_sharpe(r, 10, t, min_obs=20)
    b = trailing_sharpe(r[:t], 10, t, min_obs=20)
    assert a == b or (math.isnan(a) and math.isnan(b))


def test_pick_leader_ties_incumbent() -> None:
    scores = np.array([1.0, 1.0, 0.0])
    assert pick_leader(scores, incumbent=1) == 1
    assert pick_leader(scores, incumbent=None) == 0


def test_pick_leader_all_nan() -> None:
    assert pick_leader(np.array([np.nan, np.nan]), incumbent=0) is None


def test_onehot() -> None:
    w = onehot(4, 2)
    np.testing.assert_allclose(w, [0, 0, 1, 0])
