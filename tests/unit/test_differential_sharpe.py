"""TDD: Moody differential Sharpe helper."""
from __future__ import annotations

import math

import pytest
from tests.conftest import FLOAT_TOL

from mascotrl.eval.differential_sharpe import DifferentialSharpe


def test_differential_sharpe_finite_on_stream() -> None:
    ds = DifferentialSharpe(eta=0.05)
    rewards = []
    for r in [0.01, -0.005, 0.02, 0.0, 0.015, -0.01]:
        rewards.append(ds.step(r))
    assert all(math.isfinite(x) for x in rewards)
    assert ds.n == 6


def test_differential_sharpe_nan_input_zero() -> None:
    ds = DifferentialSharpe()
    assert ds.step(float("nan")) == pytest.approx(0.0, **FLOAT_TOL)


def test_differential_sharpe_moody_two_step_fixture() -> None:
    """Hand-check Moody (2001) EMA moments A_t, B_t and DSR at step 3 (eta=0.5).

    Returns [0.04, -0.02, 0.06]: step 1 seeds A/B; step 2 has zero variance
    (denom<=1e-12); step 3 yields finite DSR with updated moments.
    """
    eta = 0.5
    ds = DifferentialSharpe(eta=eta)
    r1, r2, r3 = 0.04, -0.02, 0.06

    assert ds.step(r1) == pytest.approx(0.0, **FLOAT_TOL)
    assert ds.A == pytest.approx(r1)
    assert ds.B == pytest.approx(r1 * r1)

    assert ds.step(r2) == pytest.approx(0.0, **FLOAT_TOL)
    assert ds.A == pytest.approx(0.01)
    assert ds.B == pytest.approx(0.001)

    dsr3 = ds.step(r3)
    assert dsr3 == pytest.approx(1.37037037037037)
    assert ds.A == pytest.approx(0.035)
    assert ds.B == pytest.approx(0.0023)
