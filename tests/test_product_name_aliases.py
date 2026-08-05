"""Plain-name claim category aliases and glossary."""
from __future__ import annotations

from mascotrl.reporting.claim_language import (
    CLAIM_CATEGORY_DH_OPTION_ALLOCATOR,
    CLAIM_CATEGORY_RANK1,
    stamp_dh_option_allocator_claim_category,
    stamp_rank1_claim_category,
)
from mascotrl.reporting.claim_stamps import (
    apply_dh_option_allocator_claim_gates,
    apply_rank1_claim_gates,
)


def test_allocator_category_alias_equals_legacy() -> None:
    assert CLAIM_CATEGORY_DH_OPTION_ALLOCATOR == "constrained_dh_panel_allocation_cmdp"
    assert CLAIM_CATEGORY_RANK1 is CLAIM_CATEGORY_DH_OPTION_ALLOCATOR


def test_stamp_allocator_default_and_legacy_alias() -> None:
    out = stamp_dh_option_allocator_claim_category({})
    assert out["claim_category"] == CLAIM_CATEGORY_DH_OPTION_ALLOCATOR
    legacy = stamp_rank1_claim_category({})
    assert legacy["claim_category"] == CLAIM_CATEGORY_DH_OPTION_ALLOCATOR
    rewritten = stamp_dh_option_allocator_claim_category(
        {"claim_category": "deep_hedge_mdp"}
    )
    assert rewritten["claim_category"] == CLAIM_CATEGORY_DH_OPTION_ALLOCATOR


def test_apply_gates_alias_is_same_callable() -> None:
    assert apply_rank1_claim_gates is apply_dh_option_allocator_claim_gates
