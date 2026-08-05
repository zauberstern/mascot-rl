"""Toy / dry panels must not seal research_positive_alpha as OM evidence."""
from __future__ import annotations

from mascotrl.reporting.claim_stamps import stamp_research_positive_alpha


def _green(**kw):
    base = {
        "train_objective_equals_claim_metric": True,
        "friction_applied": True,
        "headline_fill": "pct75",
        "fill_ladder": {"mid": 0.1, "pct75": 0.2, "worst": -0.1},
        "path_summary": {"sharpe_mean": 0.5},
        "random_baseline_sharpe": 0.0,
        "panel_source": "optionmetrics",
    }
    base.update(kw)
    return base


def test_toy_panel_refuses_research_positive_seal() -> None:
    out = stamp_research_positive_alpha(_green(panel_source="toy"))
    assert out["research_positive_alpha"] is False
    assert any("toy" in f for f in (out.get("research_positive_failures") or []))


def test_dry_run_flag_refuses_seal() -> None:
    out = stamp_research_positive_alpha(_green(dry_run=True))
    assert out["research_positive_alpha"] is False


def test_om_panel_can_pass_when_predicates_hold() -> None:
    out = stamp_research_positive_alpha(_green(panel_source="optionmetrics"))
    assert out["research_positive_alpha"] is True
