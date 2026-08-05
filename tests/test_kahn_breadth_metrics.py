"""Kahn breadth metrics tests."""
from __future__ import annotations

import pytest
from tests.conftest import FLOAT_TOL

import numpy as np

from mascotrl.eval.kahn_breadth import (
    effective_breadth,
    effective_number_of_bets_entropy,
    kahn_ir_components,
    kahn_pack,
    signal_refresh_rate,
    turnover_normalized_mean,
)
from mascotrl.reporting.capital_gates import PROJECTION_K_CEILING


def test_effective_breadth_perfect_corr() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=100)
    x = np.column_stack([base, base, base])
    neff = effective_breadth(x)
    assert neff < 2.0


def test_effective_breadth_independent() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 10))
    neff = effective_breadth(x)
    assert neff > 5.0


def test_k_scale_refused_when_alpha_nonpositive() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 10))
    pack = kahn_pack(
        x,
        rng.normal(size=50),
        np.abs(rng.normal(size=50)) * 0.1,
        factor_alpha_positive=False,
        saturation_flag=False,
        k=10,
        projection_ceiling=PROJECTION_K_CEILING,
    )
    assert pack["k_scale_claim_allowed"] is False


def test_turnover_normalized_mean() -> None:
    assert abs(turnover_normalized_mean(np.array([0.02, 0.02]), np.array([0.1, 0.1])) - 0.2) < 1e-9


def test_effective_number_of_bets_entropy_identity_vs_rank1() -> None:
    identity = np.eye(5)
    enb_id = effective_number_of_bets_entropy(identity)
    assert abs(enb_id - 5.0) < 1e-9

    # Rank-1 correlation: all ones (perfectly correlated).
    ones = np.ones((5, 5))
    enb_r1 = effective_number_of_bets_entropy(ones)
    assert enb_r1 < 1.5


def test_kahn_ir_components_fundamental_law() -> None:
    out = kahn_ir_components(ic=0.05, n_eff=25.0, g_refresh=12.0, tc=0.8)
    br = 12.0 * 25.0
    expected = 0.05 * np.sqrt(br) * 0.8
    assert abs(out["predicted_ir"] - expected) < 1e-12
    assert abs(out["breadth"] - br) < 1e-12
    assert out["ic"] == pytest.approx(0.05, **FLOAT_TOL)
    assert out["tc"] == pytest.approx(0.8, **FLOAT_TOL)
    assert out["n_eff"] == pytest.approx(25.0, **FLOAT_TOL)
    assert out["g_refresh"] == pytest.approx(12.0, **FLOAT_TOL)


def test_signal_refresh_rate_persistent_vs_noise() -> None:
    rng = np.random.default_rng(3)
    t, k = 80, 20
    # Persistent forecasts: AR(1) ~ 0.9
    persistent = np.zeros((t, k))
    persistent[0] = rng.normal(size=k)
    for i in range(1, t):
        persistent[i] = 0.9 * persistent[i - 1] + 0.1 * rng.normal(size=k)
    g_slow = signal_refresh_rate(persistent, periods_per_year=252.0)
    # White-noise forecasts: low serial dependence → higher refresh.
    noisy = rng.normal(size=(t, k))
    g_fast = signal_refresh_rate(noisy, periods_per_year=252.0)
    assert np.isfinite(g_slow) and np.isfinite(g_fast)
    assert g_fast > g_slow


def test_kahn_pack_includes_enb_when_real_returns() -> None:
    rng = np.random.default_rng(4)
    returns = rng.normal(size=(100, 8))
    pnls = rng.normal(size=100) * 0.01
    turns = np.abs(rng.normal(size=100)) * 0.05
    pack = kahn_pack(
        returns,
        pnls,
        turns,
        ic_after_cost=0.02,
        factor_alpha_positive=True,
        saturation_flag=False,
        k=8,
        g_refresh=12.0,
        tc=0.7,
    )
    assert np.isfinite(pack["effective_breadth"])
    assert np.isfinite(pack["n_eff_enb"])
    assert pack["n_eff_enb"] > 1.0
    assert "predicted_ir" in pack
    assert np.isfinite(pack["predicted_ir"])
    assert pack.get("status") != "refused_until_panel_returns"
