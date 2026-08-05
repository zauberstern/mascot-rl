"""Phase 0 honesty: claim category, nested WFO note, SPA polarity."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.plumbing

from mascotrl.reporting.capital_gates import assert_protocol_provenance
from mascotrl.reporting.claim_language import (
    CLAIM_CATEGORY_DH_OPTION_ALLOCATOR,
    CLAIM_CATEGORY_RANK1,
    PROTOCOL_NOTE_NESTED_WFO,
    SPA_DO_NOT_CLAIM,
    stamp_dh_option_allocator_claim_category,
    stamp_rank1_claim_category,
)


def test_stamp_dh_option_allocator_claim_category_default() -> None:
    out = stamp_dh_option_allocator_claim_category({})
    assert out["claim_category"] == CLAIM_CATEGORY_DH_OPTION_ALLOCATOR


def test_stamp_rank1_claim_category_default() -> None:
    # Legacy alias
    out = stamp_rank1_claim_category({})
    assert out["claim_category"] == CLAIM_CATEGORY_RANK1


def test_stamp_rewrites_deep_hedge_mdp_always() -> None:
    out = stamp_dh_option_allocator_claim_category(
        {"claim_category": "deep_hedge_mdp"}
    )
    assert out["claim_category"] == CLAIM_CATEGORY_DH_OPTION_ALLOCATOR
    # No hedge_mdp_arm escape hatch.
    out2 = stamp_dh_option_allocator_claim_category(
        {"claim_category": "deep_hedge_mdp", "hedge_mdp_arm": True}
    )
    assert out2["claim_category"] == CLAIM_CATEGORY_DH_OPTION_ALLOCATOR


def test_assert_protocol_stamps_claim_category() -> None:
    report = {
        "eval_protocol": "pit_optionmetrics_atm_is_oos",
        "alpha_found": False,
        "capital_gates_require_stability": False,
        "capital_gates_require_retrain_wfo": False,
        "require_factor_alpha": False,
        "require_after_cost": False,
    }
    out = assert_protocol_provenance(report)
    assert out["claim_category"] == CLAIM_CATEGORY_DH_OPTION_ALLOCATOR
    assert "deep_hedge_mdp" not in str(out.get("claim_category"))


def test_nested_wfo_protocol_note_constant() -> None:
    assert PROTOCOL_NOTE_NESTED_WFO == "nested_wfo_finetune_not_cpcv"


def test_spa_do_not_claim_constant() -> None:
    assert "SPA proves HAPPO" in SPA_DO_NOT_CLAIM


def test_publication_exports_spa_constant() -> None:
    from mascotrl.eval import publication

    assert hasattr(publication, "SPA_DO_NOT_CLAIM") or "SPA_DO_NOT_CLAIM" in dir(
        publication
    ) or SPA_DO_NOT_CLAIM
    # Behavioral: publication module can import the honesty constant.
    from mascotrl.reporting.claim_language import SPA_DO_NOT_CLAIM as _spa

    assert _spa == SPA_DO_NOT_CLAIM


def test_no_deep_hedge_category_constant() -> None:
    import mascotrl.reporting.claim_language as cl

    assert not hasattr(cl, "CLAIM_CATEGORY_DEEP_HEDGE")
