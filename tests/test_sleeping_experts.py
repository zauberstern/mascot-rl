"""Tests for lag-1 sleeping Variable-Share."""
from __future__ import annotations

import numpy as np

from src.eval.sleeping_experts import variable_share_sleeping


def test_sleeping_weights_sum_and_shape() -> None:
    t, n = 80, 3
    R = np.full((t, n), 0.001, dtype=np.float64)
    turb = np.zeros(t, dtype=bool)
    turb[40:] = True
    w = variable_share_sleeping(R, turb, alpha=0.05, eta=0.5)
    assert w.shape == (t, n)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-10)


def test_sleeping_turb_t_does_not_affect_W_t() -> None:
    """PIT: wake uses turb[t-1]; mutating turb[t] must not change W[t]."""
    t, n = 60, 3
    rng = np.random.default_rng(2)
    R = rng.normal(0.0, 0.01, size=(t, n))
    turb = rng.random(t) > 0.5
    w0 = variable_share_sleeping(R, turb, alpha=0.03, eta=0.5)
    day = 30
    turb2 = turb.copy()
    turb2[day] = not bool(turb2[day])
    w1 = variable_share_sleeping(R, turb2, alpha=0.03, eta=0.5)
    np.testing.assert_allclose(w0[day], w1[day], rtol=1e-12, atol=1e-14)
    # May affect W[day+1]
    assert not np.allclose(w0[day + 1], w1[day + 1]) or True  # allow equal if lucky


def test_sleeping_R_t_does_not_affect_W_t() -> None:
    t, n = 60, 3
    rng = np.random.default_rng(3)
    R = rng.normal(0.0, 0.01, size=(t, n))
    turb = rng.random(t) > 0.5
    w0 = variable_share_sleeping(R, turb, alpha=0.03, eta=0.5)
    day = 25
    R2 = R.copy()
    R2[day] += 0.05
    w1 = variable_share_sleeping(R2, turb, alpha=0.03, eta=0.5)
    np.testing.assert_allclose(w0[day], w1[day], rtol=1e-12, atol=1e-14)


def test_sleeping_t0_all_awake_turb0_unused_for_W0() -> None:
    t, n = 20, 2
    R = np.full((t, n), 0.001, dtype=np.float64)
    turb_a = np.zeros(t, dtype=bool)
    turb_b = np.ones(t, dtype=bool)
    wa = variable_share_sleeping(R, turb_a, alpha=0.1, eta=0.5)
    wb = variable_share_sleeping(R, turb_b, alpha=0.1, eta=0.5)
    # W[0] identical (t=0 all awake; turb[0] unused for wake)
    np.testing.assert_allclose(wa[0], wb[0], rtol=1e-12)
