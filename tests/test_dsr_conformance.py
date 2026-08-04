"""Differential Sharpe conformance vs Moody & Saffell Eq. 13."""
from __future__ import annotations

import math

import pytest
from tests.conftest import FLOAT_TOL

from src.eval.differential_sharpe import DifferentialSharpe


def _moody_dt(A: float, B: float, x: float) -> float:
    dA = x - A
    dB = x * x - B
    denom = B - A * A
    if denom <= 1e-12:
        return 0.0
    return (B * dA - 0.5 * A * dB) / (denom ** 1.5)


def test_dsr_matches_moody_equation():
    ds = DifferentialSharpe(eta=0.05)
    stream = [0.01, -0.02, 0.015, 0.0, 0.03, -0.01]
    A, B = 0.0, 0.0
    for i, r in enumerate(stream):
        if i == 0:
            assert ds.step(r) == pytest.approx(0.0, **FLOAT_TOL)
            A, B = r, r * r
            continue
        expected = _moody_dt(A, B, r)
        got = ds.step(r)
        assert got == pytest.approx(expected, rel=1e-9, abs=1e-12)
        A = A + ds.eta * (r - A)
        B = B + ds.eta * (r * r - B)
        # After step, moments match
        assert ds.A == pytest.approx(A)
        assert ds.B == pytest.approx(B)


def test_dsr_first_obs_zero_not_nan():
    ds = DifferentialSharpe(eta=0.01)
    assert ds.step(0.05) == pytest.approx(0.0, **FLOAT_TOL)
    assert math.isfinite(ds.A) and math.isfinite(ds.B)


def test_dsr_constant_returns_near_zero():
    ds = DifferentialSharpe(eta=0.1)
    out = [ds.step(0.01) for _ in range(50)]
    assert out[0] == pytest.approx(0.0, **FLOAT_TOL)
    assert all(abs(x) < 1e-6 for x in out[1:])


def test_dsr_single_spike_bounded():
    ds = DifferentialSharpe(eta=0.01)
    ds.step(0.0)
    ds.step(0.0)
    spike = ds.step(1.0)
    assert math.isfinite(spike)
    assert abs(spike) < 1e6


def test_dsr_small_denom_returns_zero():
    ds = DifferentialSharpe(eta=0.01)
    ds.A = 1.0
    ds.B = 1.0 + 1e-15  # denom ~ 0
    ds.n = 2
    assert ds.step(1.0) == pytest.approx(0.0, **FLOAT_TOL)


def test_dsr_nan_inf_input_unchanged_moments():
    ds = DifferentialSharpe(eta=0.01)
    ds.step(0.02)
    ds.step(-0.01)
    a0, b0, n0 = ds.A, ds.B, ds.n
    assert ds.step(float("nan")) == pytest.approx(0.0, **FLOAT_TOL)
    assert ds.A == a0 and ds.B == b0 and ds.n == n0
    assert ds.step(float("inf")) == pytest.approx(0.0, **FLOAT_TOL)
    assert ds.A == a0 and ds.B == b0 and ds.n == n0


def test_dsr_finite_difference_matches_sharpe_increment():
    """D_t ≈ (S(A+ηΔA, B+ηΔB) - S(A,B)) / η for small η (Moody identity)."""
    eta = 1e-4
    A, B = 0.01, 0.0005
    x = 0.02
    dA = x - A
    dB = x * x - B

    def sharpe(a: float, b: float) -> float:
        v = b - a * a
        if v <= 1e-12:
            return 0.0
        return a / math.sqrt(v)

    s0 = sharpe(A, B)
    s1 = sharpe(A + eta * dA, B + eta * dB)
    fd = (s1 - s0) / eta
    analytic = _moody_dt(A, B, x)
    assert analytic == pytest.approx(fd, rel=1e-3, abs=1e-5)
