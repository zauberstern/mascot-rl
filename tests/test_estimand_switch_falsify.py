"""TDD: eq OM fail → refuse alpha; archived hedge null never allocator DH alpha."""
from __future__ import annotations

import json
from pathlib import Path

from src.reporting.claim_stamps import (
    CLAIM_CATEGORY_EQ_STK,
    CLAIM_CATEGORY_RANK1,
    stamp_research_positive_alpha,
)

ROOT = Path(__file__).resolve().parents[1]
EQ_ART = (
    ROOT
    / "logs"
    / "artifacts"
    / "research_alpha"
    / "archive_kill"
    / "cpcv_eq_stk.json"
)
HEDGE_ART = (
    ROOT
    / "archive"
    / "sealed_nulls"
    / "hedge_mdp_20260807"
    / "cpcv_hedge_gate1_om_beef_tier_s.json"
)


def test_eq_om_artifact_refuses_research_alpha_if_present() -> None:
    if not EQ_ART.is_file():
        return  # optional sealed artifact; unit path covered elsewhere
    art = json.loads(EQ_ART.read_text(encoding="utf-8"))
    assert art.get("panel_source") == "optionmetrics"
    assert art.get("claim_category") == CLAIM_CATEGORY_EQ_STK
    assert art.get("claim_label_stem") == "stk_ret"
    assert art.get("research_positive_alpha") is False
    # Re-stamp must stay false.
    stamped = stamp_research_positive_alpha(art)
    assert stamped["research_positive_alpha"] is False


def test_archived_hedge_art_never_sets_allocator_alpha() -> None:
    if not HEDGE_ART.is_file():
        return
    art = json.loads(HEDGE_ART.read_text(encoding="utf-8"))
    assert art.get("gate1_pass") is False
    out = stamp_research_positive_alpha(
        {
            **art,
            "claim_category": "deep_hedge_mdp",
            "hedge_mdp_arm": True,
            "train_objective_equals_claim_metric": True,
            "friction_applied": True,
            "headline_fill": "pct75",
            "fill_ladder": {"mid": 0.1, "pct75": 0.5, "worst": 0.0},
            "path_summary": {"sharpe_mean": 0.5},
            "random_baseline_sharpe": 0.0,
            "panel_source": "optionmetrics",
        }
    )
    assert out["research_positive_alpha"] is False


def test_falsify_never_promotes_rank1_dh_as_alpha() -> None:
    out = stamp_research_positive_alpha(
        {
            "claim_category": CLAIM_CATEGORY_RANK1,
            "claim_label_stem": "dh_ret_lagdelta",
            "train_objective_equals_claim_metric": True,
            "friction_applied": True,
            "headline_fill": "pct75",
            "fill_ladder": {"mid": 0.1, "pct75": 0.5, "worst": 0.0},
            "path_summary": {"sharpe_mean": 0.5},
            "random_baseline_sharpe": 0.0,
            "panel_source": "optionmetrics",
        }
    )
    # Eq victory path is a different category; delta-hedged option allocator stem must not satisfy eq gates.
    eqish = stamp_research_positive_alpha(
        {
            "claim_category": CLAIM_CATEGORY_RANK1,
            "claim_label_stem": "stk_ret",
            "train_objective_equals_claim_metric": True,
            "friction_applied": True,
            "headline_fill": "pct75",
            "fill_ladder": {"mid": 0.1, "pct75": 0.5, "worst": 0.0},
            "path_summary": {"sharpe_mean": 0.5},
            "random_baseline_sharpe": 0.0,
            "sign_lag_baseline_sharpe": -0.1,
            "long_baseline_sharpe": 0.0,
            "panel_source": "optionmetrics",
        }
    )
    assert eqish["research_positive_alpha"] is False
    fails = eqish.get("research_positive_failures") or []
    assert any("claim_category" in f for f in fails)
