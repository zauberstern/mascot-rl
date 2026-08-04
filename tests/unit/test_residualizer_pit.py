"""Step 12: fold-fitted residualizers never peek at post-train factors."""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.residualization import (
    ResidualizerState,
    fit_ff4_residualizer,
    fit_ipca3_residualizer,
    freeze_residualizer,
    residualize_step,
)


def test_fit_ff4_accepts_wider_gate2_panel():
    """equity_substrate may emit 7 factors; FF4 residualization uses mkt/smb/hml/umd."""
    rng = np.random.default_rng(1)
    T = 60
    # Layout: mkt,smb,hml,rmw,cma,umd,ps
    X7 = rng.normal(size=(T, 7))
    true_b = np.array([0.4, -0.1, 0.2, 0.15], dtype=np.float64)
    y = X7[:, [0, 1, 2, 5]] @ true_b + rng.normal(scale=0.01, size=T)
    state = fit_ff4_residualizer(y, X7, fold_id="wide")
    assert state.betas.shape == (4,)
    np.testing.assert_allclose(state.betas, true_b, atol=0.08)


def test_fit_ff4_ols_on_train_arrays_only():
    rng = np.random.default_rng(0)
    T = 80
    X = rng.normal(size=(T, 4))
    true_b = np.array([0.5, -0.2, 0.1, 0.05], dtype=np.float64)
    y = X @ true_b + rng.normal(scale=0.01, size=T)

    # Contaminate a held-out block that must not affect betas.
    X_full = np.vstack([X, rng.normal(size=(20, 4))])
    y_full = np.concatenate([y, rng.normal(size=20)])

    state = fit_ff4_residualizer(y, X, fold_id="f0")
    assert isinstance(state, ResidualizerState)
    assert state.model == "ff4"
    assert state.fold_id == "f0"
    assert state.betas.shape == (4,)
    assert state.factor_names == ("mkt", "smb", "hml", "mom")
    np.testing.assert_allclose(state.betas, true_b, atol=0.05)

    leaked = fit_ff4_residualizer(y_full, X_full, fold_id="leaky")
    # Train-only fit must differ from full-sample once OOS noise is material.
    assert not np.allclose(state.betas, leaked.betas)


def test_fit_ipca3_fold_frozen_pca_loadings():
    rng = np.random.default_rng(1)
    T, N = 60, 12
    # Low-rank panel: 3 latent factors.
    F = rng.normal(size=(T, 3))
    L = rng.normal(size=(N, 3))
    panel = F @ L.T + rng.normal(scale=0.01, size=(T, N))

    state = fit_ipca3_residualizer(panel, fold_id="ipca-fold")
    assert state.model == "ipca3"
    assert state.fold_id == "ipca-fold"
    assert state.factor_names == ("ipca1", "ipca2", "ipca3")
    assert state.betas.ndim >= 1
    assert state.betas.size >= 3


def test_fit_ipca3_characteristic_path_fold_frozen():
    rng = np.random.default_rng(2)
    T, N, Ldim = 40, 10, 4
    char = rng.normal(size=(T, N, Ldim))
    panel = rng.normal(scale=0.02, size=(T, N))
    state = fit_ipca3_residualizer(
        panel, characteristics=char, fold_id="ipca-char", n_iter=2
    )
    assert state.model == "ipca3"
    assert state.betas.shape == (N, 3)
    frozen = freeze_residualizer(state, "ipca-char")
    assert frozen.fold_id == "ipca-char"


def test_residualize_step_identity():
    gross = 0.012
    costs = 0.003
    exp = np.array([0.4, -0.1, 0.05, 0.02])
    fac = np.array([0.01, 0.02, -0.005, 0.001])
    expected = gross - costs - float(np.dot(exp, fac))
    got = residualize_step(gross, costs, exp, fac)
    assert got == pytest.approx(expected, rel=0, abs=0.0)


def test_freeze_residualizer_immutable_copy():
    y = np.linspace(0.0, 1.0, 20)
    X = np.column_stack([y, y**2, np.ones(20), np.arange(20) * 0.01])
    state = fit_ff4_residualizer(y, X, fold_id="train")
    frozen = freeze_residualizer(state, "frozen-fold")
    assert frozen.fold_id == "frozen-fold"
    assert frozen.model == state.model
    np.testing.assert_array_equal(frozen.betas, state.betas)
    frozen.betas[0] = 999.0
    assert state.betas[0] != 999.0
