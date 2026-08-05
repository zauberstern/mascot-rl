"""Part E.4: measured archetype scoring (frozen weights, margin rule)."""
from __future__ import annotations

import numpy as np
import pytest

from mascotrl.reporting.policy_behavior import (
    ARCHETYPE_IDS,
    ARCHETYPE_SCORE_WEIGHTS,
    assign_archetype,
    score_archetypes,
)


# Frozen a-priori table (reduced 6-archetype panel: 5 named + mixed fallback).
_LEDGER_WEIGHTS = {
    "trend_follower": {
        "tilt_trend": 0.5,
        "tilt_autocorr_lag21": 0.2,
        "holding_period_days": 0.2,
        "tilt_reversal": -0.1,
    },
    "contrarian": {
        "tilt_reversal": 0.5,
        "turnover_mean": 0.3,
        "tilt_trend": -0.2,
    },
    "risk_manager": {
        "tilt_defensive": 0.4,
        "neg_downside_capture": 0.3,
        "neg_max_weight_mean": 0.2,
        "b_defensive_vix": 0.1,
    },
    "speculator": {
        "tilt_lottery": 0.4,
        "hhi_mean": 0.3,
        "max_weight_mean": 0.2,
        "upside_capture": 0.1,
    },
    "tactical_rotator": {
        "rotation_rate": 0.5,
        "across_regime_tilt_variance": 0.3,
        "within_regime_tilt_variance": -0.2,
    },
}


def _base_row(**overrides):
    row = {
        "tilt_trend": 0.0,
        "tilt_reversal": 0.0,
        "tilt_carry": 0.0,
        "tilt_defensive": 0.0,
        "tilt_lottery": 0.0,
        "tilt_illiquid": 0.0,
        "tilt_core": 0.0,
        "tilt_autocorr_lag21": 0.0,
        "holding_period_days": 10.0,
        "turnover_mean": 0.1,
        "hhi_mean": 0.2,
        "max_weight_mean": 0.3,
        "l1_vs_ew_mean": 0.2,
        "rotation_rate": 0.1,
        "return_skew": 0.0,
        "downside_capture": 1.0,
        "upside_capture": 1.0,
        "b_defensive_vix": 0.0,
        "across_regime_tilt_variance": 0.1,
        "within_regime_tilt_variance": 0.1,
    }
    row.update(overrides)
    return row


def test_frozen_weights_match_trial_ledger_entry():
    assert ARCHETYPE_SCORE_WEIGHTS == _LEDGER_WEIGHTS
    assert set(ARCHETYPE_SCORE_WEIGHTS) == set(ARCHETYPE_IDS)


def test_trend_tilted_path_scores_trend_follower():
    panel = [
        _base_row(tilt_trend=0.8, tilt_autocorr_lag21=0.9, holding_period_days=40, tilt_reversal=-0.2),
        _base_row(tilt_reversal=0.7, turnover_mean=0.5),
        _base_row(tilt_lottery=0.8, hhi_mean=0.9, max_weight_mean=0.8),
        _base_row(l1_vs_ew_mean=0.01, hhi_mean=0.11, rotation_rate=0.01),
    ]
    scores = score_archetypes(panel)
    assert set(scores[0]) == set(ARCHETYPE_IDS)
    assert max(scores[0], key=scores[0].get) == "trend_follower"


def test_equal_weight_like_row_falls_to_mixed():
    """Near-EW rows no longer have index_hugger; assignment falls to mixed."""
    panel = [
        _base_row(l1_vs_ew_mean=0.0, hhi_mean=0.1, rotation_rate=0.0),
        _base_row(tilt_trend=0.9, tilt_autocorr_lag21=0.8, holding_period_days=50),
        _base_row(tilt_lottery=0.9, hhi_mean=0.95, max_weight_mean=0.9),
        _base_row(rotation_rate=0.9, across_regime_tilt_variance=0.8),
    ]
    scores = score_archetypes(panel)
    decision = assign_archetype(scores[0])
    # Without index_hugger, a near-EW row should not clear a named archetype margin.
    assert decision["archetype_primary"] == "mixed" or decision["archetype_margin"] < 0.25


def test_lottery_concentrated_scores_speculator():
    panel = [
        _base_row(tilt_lottery=0.9, hhi_mean=0.95, max_weight_mean=0.9, upside_capture=1.5),
        _base_row(tilt_trend=0.8, tilt_autocorr_lag21=0.7, holding_period_days=40),
        _base_row(l1_vs_ew_mean=0.0, hhi_mean=0.1, rotation_rate=0.0),
        _base_row(tilt_carry=0.8, holding_period_days=30, return_skew=-0.5),
    ]
    scores = score_archetypes(panel)
    assert max(scores[0], key=scores[0].get) == "speculator"


def test_tie_within_margin_returns_mixed():
    # Two rows nearly collinear on trend vs contrarian levers → small margin
    a = _base_row(tilt_trend=0.5, tilt_reversal=0.5, turnover_mean=0.3, tilt_autocorr_lag21=0.5)
    b = _base_row(tilt_trend=0.0, tilt_reversal=0.0, turnover_mean=0.1)
    c = _base_row(tilt_trend=1.0, tilt_reversal=0.0, holding_period_days=50)
    d = _base_row(tilt_trend=0.0, tilt_reversal=1.0, turnover_mean=0.6)
    scores = score_archetypes([a, b, c, d])
    # Force a near-tie by constructing equal top scores directly
    tied = {k: 0.0 for k in ARCHETYPE_IDS}
    tied["trend_follower"] = 1.0
    tied["contrarian"] = 0.9  # margin 0.1 < 0.25
    decision = assign_archetype(tied)
    assert decision["archetype_primary"] == "mixed"
    assert decision["archetype_runner_up"] in ("trend_follower", "contrarian")
    assert decision["archetype_margin"] == pytest.approx(0.1, abs=1e-12)


def test_full_score_vector_always_present():
    panel = [_base_row(), _base_row(tilt_trend=0.3), _base_row(tilt_lottery=0.4)]
    for s in score_archetypes(panel):
        assert set(s) == set(ARCHETYPE_IDS)
        assert all(isinstance(v, float) for v in s.values())
