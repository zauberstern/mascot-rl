"""Claim flags (estimand, transfer, arm-lock, research).

Hire: claim = w · dh_ret_lagdelta on OM ATM.
Fire: soft train fees as institutional claim; EstimandSpec as overnight SoT;
      full OM hedge ART merged into allocator pack.
Perfect: thin flags + OM-touch cost-in-decision on hist/CPCV finetune only.
Research carve-out: stamp_research_positive_alpha (never unlocks institutional).
"""
from __future__ import annotations

from typing import Any

from src.data.oos_panel import LABEL_STEM
from src.reporting.claim_language import (
    CLAIM_CATEGORY_DH_OPTION_ALLOCATOR,
    stamp_dh_option_allocator_claim_category,
)

_DEEP_HEDGE_MDP_LEGACY = "deep_hedge_mdp"

CLAIM_RETURN_DEFINITION = "delta_hedged_call_lagdelta_scaled_by_delta_S_minus_C"
# Default transfer label; override via report/cfg transfer_protocol.
TRANSFER_PROTOCOL = "rbergomi_dupire_pretrain_then_optionmetrics_finetune"
TRANSFER_PROTOCOL_BY_WORLD = {
    "historical": "optionmetrics_hist_train_then_optionmetrics_eval",
    "rbergomi": "rbergomi_dupire_pretrain_then_optionmetrics_finetune",
    "gbm": "gbm_pretrain_then_optionmetrics_eval",
    "heston": "heston_pretrain_then_optionmetrics_eval",
    "garch": "garch_pretrain_then_optionmetrics_eval",
    "hybrid_pretrain_finetune": "sim_pretrain_then_optionmetrics_finetune",
}
# Frozen campaign id string (legacy value kept for ART compatibility).
CAMPAIGN_DH_OPTION_ALLOCATOR = "rank1_allocator"
CAMPAIGN_RANK1 = CAMPAIGN_DH_OPTION_ALLOCATOR
CLAIM_CATEGORY_EQ_STK = "equity_stk_ret_allocation"

# Re-export allocator category for callers that import from this module.
CLAIM_CATEGORY_RANK1 = CLAIM_CATEGORY_DH_OPTION_ALLOCATOR


def stamp_dh_option_allocator_estimand_and_transfer(
    report: dict[str, Any],
) -> dict[str, Any]:
    """Stamp allocator estimand identity + transfer protocol (honest train≠claim)."""
    out = stamp_dh_option_allocator_claim_category(dict(report))
    out["campaign"] = str(out.get("campaign") or CAMPAIGN_DH_OPTION_ALLOCATOR)
    ho = out.get("historical_oos") if isinstance(out.get("historical_oos"), dict) else {}

    stem = str(ho.get("label_stem") or out.get("claim_label_stem") or LABEL_STEM)
    ret_def = str(
        ho.get("return_definition")
        or out.get("claim_return_definition")
        or CLAIM_RETURN_DEFINITION
    )
    out["claim_label_stem"] = stem
    out["claim_return_definition"] = ret_def
    out["train_reward"] = str(out.get("train_reward") or "clean_mtm_synth")
    train_world = str(
        out.get("train_world")
        or out.get("train_distribution")
        or "rbergomi"
    )
    if train_world in ("rbergomi_dupire", "synthetic"):
        train_world = "rbergomi"
    if train_world in ("optionmetrics",):
        train_world = "historical"
    out["train_world"] = train_world
    out.setdefault("train_distribution", train_world)

    om_touch = bool(
        out.get("om_touch_enabled")
        or ((out.get("plugins") or {}).get("om_touch") or {}).get("enabled")
        or ho.get("friction_model") == "om_touch"
    )
    if om_touch or bool(ho.get("friction_applied")):
        out["eval_friction"] = (
            "om_touch"
            if om_touch or ho.get("friction_model") == "om_touch"
            else str(ho.get("friction_model") or out.get("eval_friction") or "unknown")
        )
    else:
        out["eval_friction"] = str(out.get("eval_friction") or "none")

    # Synth MTM ≠ DH claim until hist finetune is the decision path.
    out["train_objective_equals_claim_metric"] = False

    nested = out.get("nested_wfo") if isinstance(out.get("nested_wfo"), dict) else {}
    cpcv = out.get("cpcv") if isinstance(out.get("cpcv"), dict) else {}
    finetune_friction = bool(
        out.get("finetune_friction_applied")
        or nested.get("finetune_friction_applied")
        or cpcv.get("finetune_friction_applied")
    )
    out["finetune_friction_applied"] = finetune_friction
    out["transfer_protocol"] = str(
        out.get("transfer_protocol")
        or nested.get("transfer_protocol")
        or TRANSFER_PROTOCOL_BY_WORLD.get(train_world, TRANSFER_PROTOCOL)
    )
    n_folds = int(nested.get("n_folds") or cpcv.get("n_folds") or 0)
    has_fold_path = n_folds > 0 or bool(cpcv.get("folds")) or bool(nested.get("folds"))
    if "transfer_ok" in report:
        out["transfer_ok"] = bool(report.get("transfer_ok"))
    else:
        out["transfer_ok"] = bool(finetune_friction and has_fold_path)
    return out


def assert_dh_option_allocator_arm_lock(report: dict[str, Any]) -> dict[str, Any]:
    """Refuse deep-hedge / archived hedge ART merge on allocator overnight reports."""
    out = stamp_dh_option_allocator_claim_category(dict(report))
    failures: list[str] = []
    if out.get("arm") in ("hedge_mdp", "hedge", "deep_hedge"):
        failures.append("hedge_mdp_arm_out_of_scope")
        out.pop("arm", None)
    if out.get("hedge_mdp_arm"):
        failures.append("hedge_mdp_arm_out_of_scope")
        out.pop("hedge_mdp_arm", None)

    pack = out.get("publication_evidence_pack")
    if isinstance(pack, dict) and pack:
        pack_arm = str(pack.get("arm") or "")
        pack_cat = str(pack.get("claim_category") or "")
        src = str(pack.get("source") or pack.get("artifact_path") or "")
        if (
            pack_arm in ("hedge_mdp", "hedge", "deep_hedge")
            or pack_cat == _DEEP_HEDGE_MDP_LEGACY
            or "hedge_gate1" in src
            or "beef" in src.lower()
        ):
            failures.append("beef_art_in_rank1_pack")  # legacy failure id
            pack = dict(pack)
            pack.pop("arm", None)
            if pack.get("claim_category") == _DEEP_HEDGE_MDP_LEGACY:
                pack["claim_category"] = CLAIM_CATEGORY_DH_OPTION_ALLOCATOR
            pack["arm_lock_refused_beef"] = True
            out["publication_evidence_pack"] = pack

    out["campaign"] = str(out.get("campaign") or CAMPAIGN_DH_OPTION_ALLOCATOR)
    if failures:
        out["arm_lock_failures"] = failures
    return out


def apply_dh_option_allocator_claim_gates(
    report: dict[str, Any],
    failures: list[str],
) -> list[str]:
    """Stamp report in-place and append allocator estimand/friction/transfer/arm failures."""
    stamped = stamp_dh_option_allocator_estimand_and_transfer(report)
    locked = assert_dh_option_allocator_arm_lock(stamped)
    report.update(locked)

    ho = report.get("historical_oos") if isinstance(report.get("historical_oos"), dict) else {}
    require_om = bool(
        report.get("om_touch_enabled")
        or ((report.get("plugins") or {}).get("om_touch") or {}).get("enabled")
        or report.get("capital_gates_require_om_touch_claim", False)
    )
    stem = str(report.get("claim_label_stem") or ho.get("label_stem") or "")
    if stem and stem != LABEL_STEM:
        failures.append(f"claim_label_stem_mismatch (got {stem}, want {LABEL_STEM})")
    if require_om and ho and ho.get("friction_applied") is False:
        failures.append("friction_applied_false_under_om_touch_claim")

    require_retrain = bool(report.get("capital_gates_require_retrain_wfo", True))
    require_stability = bool(report.get("capital_gates_require_stability", True))
    if require_retrain and require_stability:
        if not bool(report.get("finetune_friction_applied")):
            failures.append("finetune_friction_applied_false")
        if not bool(report.get("transfer_ok")):
            failures.append("transfer_ok_false")

    for f in report.get("arm_lock_failures") or []:
        if f not in failures:
            failures.append(str(f))
    return failures


def stamp_research_positive_alpha(report: dict[str, Any]) -> dict[str, Any]:
    """Write research_positive_alpha seal (research-tier evidence only).

    Matched train=claim + friction + after-cost CPCV Sharpe > 0 and > random,
    headline pct75. Does not emit capital-allocation claim fields.
    """
    out = dict(report)
    failures: list[str] = []

    cat = str(out.get("claim_category") or "")
    if cat == _DEEP_HEDGE_MDP_LEGACY or bool(out.get("hedge_mdp_arm")):
        failures.append("research_positive_refused_hedge_arm")
    if bool(out.get("dry_run")):
        failures.append("dry_run_not_sealable")
    src = str(out.get("panel_source") or "")
    if src in ("toy", "synthetic", ""):
        if src != "optionmetrics":
            failures.append(f"panel_source_not_optionmetrics (got {src!r})")

    if not bool(out.get("train_objective_equals_claim_metric")):
        failures.append("train_objective_ne_claim_metric")
    if not bool(out.get("friction_applied")):
        failures.append("friction_not_applied")

    ladder = out.get("fill_ladder")
    if not isinstance(ladder, dict) or not ladder:
        failures.append("fill_ladder_missing")
    headline = str(out.get("headline_fill") or "")
    if headline != "pct75":
        failures.append(f"headline_fill_not_pct75 (got {headline!r})")

    path = out.get("path_summary") if isinstance(out.get("path_summary"), dict) else {}
    try:
        sharpe = float(path.get("sharpe_mean"))
    except (TypeError, ValueError):
        sharpe = float("nan")
    if not (sharpe == sharpe) or sharpe <= 0.0:
        failures.append(f"sharpe_mean_not_positive (got {sharpe})")

    raw_rand = out.get("random_baseline_sharpe")
    if raw_rand is None:
        failures.append("random_baseline_sharpe_missing")
        rand = float("nan")
    else:
        try:
            rand = float(raw_rand)
        except (TypeError, ValueError):
            rand = float("nan")
            failures.append("random_baseline_sharpe_invalid")
    if raw_rand is not None and (not (rand == rand) or not (sharpe == sharpe) or sharpe <= rand):
        failures.append(f"sharpe_not_above_random (sharpe={sharpe}, random={rand})")

    stem = str(out.get("claim_label_stem") or "")
    cat = str(out.get("claim_category") or "")
    if cat == CLAIM_CATEGORY_EQ_STK or stem == "stk_ret":
        if cat != CLAIM_CATEGORY_EQ_STK:
            failures.append(f"claim_category_mismatch_for_stk_ret (got {cat!r})")
        if stem != "stk_ret":
            failures.append(f"claim_label_stem_not_stk_ret (got {stem!r})")
        for key, label in (
            ("sign_lag_baseline_sharpe", "sign_lag"),
            ("long_baseline_sharpe", "long"),
        ):
            raw = out.get(key)
            if raw is None:
                failures.append(f"{key}_missing")
                continue
            try:
                peer = float(raw)
            except (TypeError, ValueError):
                failures.append(f"{key}_invalid")
                continue
            if not (peer == peer) or not (sharpe == sharpe) or sharpe <= peer:
                failures.append(f"sharpe_not_above_{label} (sharpe={sharpe}, {label}={peer})")

    ok = not failures
    out["claim_tier"] = str(out.get("claim_tier") or "research")
    out["research_positive_alpha"] = bool(ok)
    if failures:
        out["research_positive_failures"] = failures
    return out


# Legacy aliases (do not use in new code).
stamp_rank1_estimand_and_transfer = stamp_dh_option_allocator_estimand_and_transfer
assert_rank1_arm_lock = assert_dh_option_allocator_arm_lock
apply_rank1_claim_gates = apply_dh_option_allocator_claim_gates
stamp_rank1_claim_category = stamp_dh_option_allocator_claim_category
