"""TDD: falsify delta-hedged option allocator research; refuse hedge-arm alpha language."""
from __future__ import annotations

from src.eval.research_alpha_baselines import policy_beats_random
from src.reporting.claim_stamps import stamp_research_positive_alpha


def test_kill_path_when_not_above_random() -> None:
    assert policy_beats_random(-0.1, 0.0) is False
    art = stamp_research_positive_alpha(
        {
            "train_objective_equals_claim_metric": True,
            "friction_applied": True,
            "headline_fill": "pct75",
            "fill_ladder": {"mid": 0.0, "pct75": -0.1, "worst": -0.5},
            "path_summary": {"sharpe_mean": -0.1},
            "random_baseline_sharpe": 0.0,
        }
    )
    assert art["research_positive_alpha"] is False


def test_hedge_arm_report_cannot_set_research_positive_alpha() -> None:
    out = stamp_research_positive_alpha(
        {
            "train_objective_equals_claim_metric": True,
            "friction_applied": True,
            "headline_fill": "pct75",
            "fill_ladder": {"mid": 0.1, "pct75": 0.5, "worst": 0.0},
            "path_summary": {"sharpe_mean": 0.5},
            "random_baseline_sharpe": 0.0,
            "panel_source": "optionmetrics",
            "claim_category": "deep_hedge_mdp",
            "hedge_mdp_arm": True,
        }
    )
    assert out["research_positive_alpha"] is False
    assert "research_positive_refused_hedge_arm" in (
        out.get("research_positive_failures") or []
    )
