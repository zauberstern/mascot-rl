"""Canonical claim-category strings for allocator reports.

Sole purpose of this module: portfolio allocation claim language for the
delta-hedged option allocator (HAPPO+CMDP on ATM ``dh_ret_lagdelta``).

Deep-hedge / hedge-MDP claim vocabulary is out of scope (Tier-1 engine deleted).
Delta-hedge language survives only as a friction cost on the option-arm stock
leg (``hedge_leg_cost`` / ``plugins/hedge_impact``), not as a product claim.

Glossary (plain names; do not invent ordinal nicknames in new prose):

| Legacy nickname | Say instead | Meaning |
|-----------------|-------------|---------|
| Rank-1 | delta-hedged option allocator | HAPPO picks weights on ATM ``dh_ret_lagdelta`` under CMDP |
| capital spine | main HAPPO training path | ``config/workflows/happo_cmdp_mamba_k50.yaml`` + train scripts |
| research_alpha_v0 | research alpha trial | Same allocator object; weaker claim (matched costs in train) |
| overnight (campaign) | overnight train-and-evaluate run | Long train + evaluate campaign scripts |

This this repo makes no capital-allocation claim fields.
"""
from __future__ import annotations

# Estimand-true string values (frozen for sealed ART compatibility).
CLAIM_CATEGORY_DH_OPTION_ALLOCATOR = "constrained_dh_panel_allocation_cmdp"

# Legacy aliases (do not use in new code).
CLAIM_CATEGORY_RANK1 = CLAIM_CATEGORY_DH_OPTION_ALLOCATOR

PROTOCOL_NOTE_NESTED_WFO = "nested_wfo_finetune_not_cpcv"

SPA_DO_NOT_CLAIM = "SPA proves HAPPO alpha / SPA ranks HAPPO best strategy"

_DEEP_HEDGE_MDP_LEGACY = "deep_hedge_mdp"


def stamp_dh_option_allocator_claim_category(report: dict) -> dict:
    """Stamp allocator claim category; rewrite any deep_hedge_mdp string.

    Always leaves ``CLAIM_CATEGORY_DH_OPTION_ALLOCATOR`` when already set or when
    category is missing. Any ``deep_hedge_mdp`` value is rewritten to the
    allocator category (no ``hedge_mdp_arm`` escape hatch).
    """
    report = dict(report)
    raw = report.get("claim_category")
    if raw is None or str(raw) == "" or str(raw) == _DEEP_HEDGE_MDP_LEGACY:
        cat = CLAIM_CATEGORY_DH_OPTION_ALLOCATOR
    else:
        cat = str(raw)
        if cat == _DEEP_HEDGE_MDP_LEGACY:
            cat = CLAIM_CATEGORY_DH_OPTION_ALLOCATOR
    report["claim_category"] = cat
    return report


# Legacy alias.
stamp_rank1_claim_category = stamp_dh_option_allocator_claim_category
