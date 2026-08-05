"""TDD: summarize_policy_diagnostics rolls weights/turnover/entropy into one report."""
from __future__ import annotations

import numpy as np

from mascotrl.eval.research_alpha_cpcv import _training_policy_diagnostics
from mascotrl.eval.policy_diagnostics import summarize_policy_diagnostics


def test_summarize_policy_diagnostics_returns_expected_keys() -> None:
    k = 4
    weights = np.full((10, k), 1.0 / k)
    out = summarize_policy_diagnostics(
        weights=weights,
        turnovers=[0.0] * 10,
        entropies=[1.0] * 10,
        log_std_mean=-0.5,
        logit_std=0.1,
    )
    for key in (
        "hhi_mean",
        "hhi_max",
        "l1_vs_ew_mean",
        "l1_vs_ew_max",
        "max_weight_mean",
        "max_weight_max",
        "turnover_mean",
        "entropy_mean",
        "log_std_mean",
        "logit_std",
        "collapse_guard",
        "equal_weight_collapse_guard",
    ):
        assert key in out


def test_summarize_policy_diagnostics_flags_ew_collapse() -> None:
    k = 4
    weights = np.full((10, k), 1.0 / k)
    out = summarize_policy_diagnostics(weights=weights, turnovers=[0.0] * 10)
    assert out["equal_weight_collapse_guard"]["collapse_detected"] is True
    assert out["equal_weight_collapse_detected"] is True


def test_summarize_policy_diagnostics_handles_missing_optional_fields() -> None:
    k = 3
    rng = np.random.default_rng(1)
    weights = rng.dirichlet(np.ones(k) * 0.5, size=8)
    out = summarize_policy_diagnostics(weights=weights)
    assert np.isnan(out["turnover_mean"])
    assert np.isnan(out["entropy_mean"])
    assert out["log_std_mean"] is None
    assert out["logit_std"] is None


def test_summarize_policy_diagnostics_accepts_list_of_weight_vectors() -> None:
    rows = [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]]
    out = summarize_policy_diagnostics(weights=rows)
    assert out["hhi_mean"] > 0.0


def test_training_policy_diagnostics_threads_fold_stats() -> None:
    folds = [
        {
            "learning_curve": [{"entropy": 1.2}, {"entropy": 1.0}],
            "train_stats": {"entropy": 0.8, "log_std_mean": -0.4},
        },
        {
            "learning_curve": [{"entropy": 0.6}],
            "train_stats": {"log_std_mean": -0.2},
        },
    ]
    out = _training_policy_diagnostics(folds)
    assert out["entropies"] == [1.2, 1.0, 0.8, 0.6]
    assert np.isclose(out["log_std_mean"], -0.3)
