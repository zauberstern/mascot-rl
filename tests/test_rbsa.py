"""Sharpe RBSA constrained style regression."""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import FLOAT_TOL


def test_fit_rbsa_recovers_known_loadings():
    from src.eval.rbsa import fit_rbsa

    rng = np.random.default_rng(0)
    t, k = 200, 3
    true = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    factors = rng.normal(0.0, 0.01, size=(t, k))
    noise = rng.normal(0.0, 1e-4, size=t)
    port = factors @ true + noise
    loadings, r2 = fit_rbsa(port, factors)
    assert loadings.shape == (k,)
    assert float(np.sum(loadings)) == pytest.approx(1.0, abs=1e-6)
    assert (loadings >= -1e-8).all()
    assert np.allclose(loadings, true, atol=0.05)
    assert 0.0 <= float(r2) <= 1.0 + 1e-8
    assert float(r2) > 0.9


def test_fit_rbsa_missing_inputs_return_nan():
    from src.eval.rbsa import fit_rbsa

    loadings, r2 = fit_rbsa(None, np.ones((10, 2)))
    assert np.isnan(r2)
    assert np.isnan(loadings).all()
