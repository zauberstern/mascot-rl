"""research_positive_alpha seal: research-tier evidence only."""
from __future__ import annotations

import math

from mascotrl.reporting.claim_stamps import stamp_research_positive_alpha


def _green_report(**overrides):
    base = {
        "claim_tier": "research",
        "train_objective_equals_claim_metric": True,
        "friction_applied": True,
        "headline_fill": "pct75",
        "fill_ladder": {"mid": -0.1, "pct75": 0.2, "worst": -0.5},
        "path_summary": {"sharpe_mean": 0.4},
        "random_baseline_sharpe": 0.1,
        "panel_source": "optionmetrics",
    }
    base.update(overrides)
    return base


def test_research_positive_all_predicates_true() -> None:
    out = stamp_research_positive_alpha(_green_report())
    assert out["research_positive_alpha"] is True
    assert out["claim_tier"] == "research"


def test_research_positive_refuses_each_missing_predicate() -> None:
    cases = [
        {"train_objective_equals_claim_metric": False},
        {"friction_applied": False},
        {"path_summary": {"sharpe_mean": -0.1}},
        {"path_summary": {"sharpe_mean": 0.05}, "random_baseline_sharpe": 0.2},
        {"headline_fill": "mid"},
        {"fill_ladder": {}},
        {"path_summary": {"sharpe_mean": float("nan")}},
        {"random_baseline_sharpe": None},
    ]
    for overrides in cases:
        out = stamp_research_positive_alpha(_green_report(**overrides))
        assert out["research_positive_alpha"] is False, overrides


def test_nan_sharpe_refuses() -> None:
    out = stamp_research_positive_alpha(
        _green_report(path_summary={"sharpe_mean": math.nan})
    )
    assert out["research_positive_alpha"] is False
