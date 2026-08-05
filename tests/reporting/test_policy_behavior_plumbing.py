"""Tests for policy_behavior data_availability plumbing."""
from __future__ import annotations

import numpy as np

from mascotrl.reporting.policy_behavior import build_policy_behavior


def test_export_populates_regime_fields_when_macro_available() -> None:
    rng = np.random.default_rng(0)
    T, K = 80, 7
    S = np.eye(K, 7)
    W = np.full((T, K), 1.0 / K)
    vix = rng.standard_normal(T) * 0.5
    hy = rng.standard_normal(T) * 0.1
    term = rng.standard_normal(T) * 0.1
    R = rng.normal(0.0, 0.01, size=(T, K))
    regimes = np.array(
        ["calm"] * 30 + ["inflationary"] * 25 + ["crisis"] * 25, dtype=object
    )

    payload = build_policy_behavior(
        algo="ppo",
        weights=W,
        asset_returns=R,
        sleeve_matrix=S,
        regimes=regimes,
        vix_z=vix,
        hy_oas_z=hy,
        term_spread=term,
        n_null_shuffles=10,
    )

    by_regime = payload["behaviour_by_regime"]
    assert by_regime
    assert set(by_regime) >= {"calm", "inflationary", "crisis"}
    avail = payload["data_availability"]
    assert avail["regimes"] is True
    assert avail["sleeves"] is True
    assert avail["macro"] is True


def test_export_fail_closed_without_macro() -> None:
    T, K = 40, 5
    W = np.full((T, K), 1.0 / K)
    S = np.eye(K, 7)

    payload = build_policy_behavior(
        algo="ppo",
        weights=W,
        sleeve_matrix=S,
        n_null_shuffles=5,
    )

    assert payload["behaviour_by_regime"] == {}
    assert payload["macro_tilt_sensitivity"] == {}
    avail = payload["data_availability"]
    assert avail["regimes"] is False
    assert avail["macro"] is False
    assert "regimes" in avail["missing_reason"]
    assert "macro" in avail["missing_reason"]


def test_sensitivities_roundtrip() -> None:
    T, K = 20, 4
    W = np.full((T, K), 0.25)
    sens = {"iv_skew_30d": 0.12, "mfis_30": 0.08}

    payload = build_policy_behavior(
        algo="ppo",
        weights=W,
        sensitivities=sens,
        n_null_shuffles=5,
    )

    assert payload["signal_sensitivities"] == sens
    assert payload["data_availability"]["sensitivities"] is True
