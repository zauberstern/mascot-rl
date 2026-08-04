"""IPCA residualizer parity: custom SVD vs sklearn TruncatedSVD."""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.residualization import fit_ipca3_residualizer


def _align_signs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Flip columns of b to maximize correlation with a."""
    out = b.copy()
    for j in range(min(a.shape[1], b.shape[1])):
        if float(np.dot(a[:, j], out[:, j])) < 0:
            out[:, j] *= -1.0
    return out


def test_sklearn_pca_backend_matches_custom_svd_up_to_sign():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    panel = rng.normal(size=(40, 8))
    custom = fit_ipca3_residualizer(panel, backend="custom")
    sk = fit_ipca3_residualizer(panel, backend="sklearn_pca")
    assert custom.betas.shape == sk.betas.shape
    aligned = _align_signs(custom.betas, sk.betas)
    assert np.allclose(custom.betas, aligned, atol=1e-6, rtol=1e-5)


def test_ipca_backend_runs_or_falls_back():
    pytest.importorskip("ipca")
    rng = np.random.default_rng(1)
    t, n, l = 30, 6, 4
    panel = rng.normal(size=(t, n))
    char = rng.normal(size=(t, n, l))
    state = fit_ipca3_residualizer(panel, char, backend="ipca", n_iter=2)
    assert state.betas.shape[0] == n
    assert state.betas.shape[1] == 3
    assert np.isfinite(state.betas).all()


def test_default_backend_is_custom():
    rng = np.random.default_rng(2)
    panel = rng.normal(size=(20, 5))
    a = fit_ipca3_residualizer(panel)
    b = fit_ipca3_residualizer(panel, backend="custom")
    assert np.allclose(a.betas, b.betas)
