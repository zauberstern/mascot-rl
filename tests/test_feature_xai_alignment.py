"""Tests for feature health, XAI quality metrics, and personality alignment."""
from __future__ import annotations

import numpy as np


def test_feature_health_report_flags_dead_channel() -> None:
    from src.eval.equity_substrate import feature_health_report

    rng = np.random.default_rng(0)
    cube = rng.normal(0, 1, size=(20, 5, 3))
    cube[:, :, 2] = np.nan  # dead channel
    report = feature_health_report(cube, channel_names=["a", "b", "dead"])
    assert report["n_channels"] == 3
    assert "dead" in report["dead_channels"]
    assert report["per_channel"]["a"]["nan_rate"] < 0.01
    assert report["per_channel"]["dead"]["dead"] is True


def test_explanation_quality_metrics_sparsity() -> None:
    from src.reporting.interpretability import explanation_quality_metrics

    attribution = {
        "groups": {
            "equity": {"l1_delta": 0.10},
            "surface": {"l1_delta": 0.05},
            "state": {"l1_delta": 0.01},
            "macro": {"l1_delta": 0.001},
        }
    }
    qm = explanation_quality_metrics(attribution)
    assert "explanation_sparsity" in qm
    assert qm["n_dominant_channels"] >= 1
    assert np.isfinite(qm["explanation_sparsity"])


def test_personality_alignment_match_and_divergence() -> None:
    from src.reporting.policy_behavior import (
        compute_personality_alignment,
        designed_personality,
    )

    assert designed_personality(
        objective="cvar_ru", algo="cppo", weight_head="softmax"
    ) == "risk_manager"
    match = compute_personality_alignment(
        "risk_manager",
        {"archetype_primary": "risk_manager", "archetype_scores": {"risk_manager": 0.9}},
    )
    assert match["match"] is True
    assert match["alignment_pass"] is True
    diverge = compute_personality_alignment(
        "trend_follower",
        {"archetype_primary": "contrarian", "archetype_scores": {"contrarian": 0.8}},
    )
    assert diverge["match"] is False
    assert diverge["divergence_explanation"]
    assert 0.0 <= diverge["alignment_score"] <= 1.0
