"""Analytical boundary tests for residualize_step."""
from __future__ import annotations

import numpy as np
import pytest

from mascotrl.eval.residualization import residualize_step
from tests.conftest import FLOAT_TOL


@pytest.mark.unit
def test_pure_factor_return_residualizes_to_zero():
    """When gross PnL equals lagged exposure · factor return, residual is zero."""
    exposures = np.array([0.5, -0.2, 0.1, 0.0], dtype=np.float64)
    factors = np.array([0.01, 0.02, -0.005, 0.003], dtype=np.float64)
    factor_pnl = float(np.dot(exposures, factors))
    r = residualize_step(factor_pnl, costs=0.0, exposures_tminus1=exposures, factor_return_t=factors)
    assert r == pytest.approx(0.0, **FLOAT_TOL)


@pytest.mark.unit
def test_zero_beta_is_identity():
    """Zero exposure leaves residual equal to gross - costs - borrow - rf."""
    exposures = np.zeros(4, dtype=np.float64)
    factors = np.array([0.01, 0.02, -0.005, 0.003], dtype=np.float64)
    r = residualize_step(
        0.04,
        costs=0.01,
        exposures_tminus1=exposures,
        factor_return_t=factors,
        borrow=0.005,
        rf=0.001,
    )
    assert r == pytest.approx(0.04 - 0.01 - 0.005 - 0.001, **FLOAT_TOL)


@pytest.mark.unit
def test_positive_alpha_survives_factor_subtraction():
    """Pure factor exposure vanishes; leftover alpha remains."""
    exposures = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    factors = np.array([0.02, 0.0, 0.0, 0.0], dtype=np.float64)
    alpha = 0.015
    gross = alpha + float(np.dot(exposures, factors))
    r = residualize_step(gross, costs=0.0, exposures_tminus1=exposures, factor_return_t=factors)
    assert r == pytest.approx(alpha, **FLOAT_TOL)
