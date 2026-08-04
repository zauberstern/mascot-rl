"""Symmetric training path helpers for spectrum portfolio arms (opt/eq/mix)."""
from __future__ import annotations

from typing import Any

from src.arms.spec import (
    EQUITY_LABEL_STEM,
    OPTION_LABEL_STEM,
    ArmSpec,
    arm_spec_from_cfg,
)
from src.data.slot_mask import coverage_masks_for_arm


def resolve_claim_label_stem(cfg: dict[str, Any]) -> str:
    """Arm-faithful single-stem label for eq/opt panel loading.

    Mix has no single stem (opt||eq concat); callers must use the mix concat
    path instead. Explicit ``claim_label_stem`` must match the arm or raise.
    """
    arm = resolve_portfolio_arm(cfg)
    if arm.id == "mix":
        raise ValueError(
            "mix arm has no single claim_label_stem; build concatenated "
            f"[{OPTION_LABEL_STEM}|{EQUITY_LABEL_STEM}] label matrix"
        )
    expected = EQUITY_LABEL_STEM if arm.id == "eq" else OPTION_LABEL_STEM
    explicit = cfg.get("claim_label_stem")
    if explicit is not None and str(explicit).strip() != "":
        stem = str(explicit).strip()
        if stem != expected:
            raise ValueError(
                f"claim_label_stem={stem!r} inconsistent with portfolio_arm="
                f"{arm.id!r} (expected {expected!r})"
            )
        return stem
    return expected


def resolve_portfolio_arm(cfg: dict[str, Any]) -> ArmSpec:
    """Resolve ArmSpec from cfg.arm or cfg.portfolio_arm shorthand."""
    if cfg.get("arm"):
        return arm_spec_from_cfg(cfg)
    pa = str(cfg.get("portfolio_arm") or "opt").lower()
    n = int(cfg.get("n_assets", 50))
    if pa == "eq":
        return ArmSpec(
            id="eq",
            option_slots=0,
            equity_slots=n,
            delta_mode="soft",
            equity_label_stem="stk_ret",
            alpha_claim=True,
        )
    if pa == "mix":
        opt_n = n // 2
        eq_n = n - opt_n
        return ArmSpec(
            id="mix",
            option_slots=opt_n,
            equity_slots=eq_n,
            delta_mode="soft",
            option_label_stem="dh_ret_lagdelta",
            equity_label_stem="stk_ret",
            alpha_claim=False,  # forced diagnostic-only
        )
    return ArmSpec(
        id="opt",
        option_slots=n,
        equity_slots=0,
        delta_mode="soft",
        option_label_stem="dh_ret_lagdelta",
        alpha_claim=True,
    )


def projection_backend_for_arm(arm: ArmSpec) -> str:
    if arm.id == "mix" or (arm.option_slots > 0 and arm.equity_slots > 0):
        return "overlay_cvxpy"
    return "cvxpy"


def arm_training_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    """Stamp arm training path (features, labels, projection, claimability)."""
    arm = resolve_portfolio_arm(cfg)
    claimable = bool(arm.alpha_claim) and arm.id != "mix"
    return {
        "arm_id": arm.id,
        "option_slots": arm.option_slots,
        "equity_slots": arm.equity_slots,
        "option_label_stem": arm.option_label_stem,
        "equity_label_stem": arm.equity_label_stem,
        "projection_backend": projection_backend_for_arm(arm),
        "alpha_claim_allowed": claimable,
        "diagnostic_only": not claimable,
        "coverage_fn": "coverage_masks_for_arm",
    }


def build_arm_coverage(
    arm: ArmSpec,
    *,
    atm=None,
    option_labels=None,
    equity_labels=None,
    spot=None,
):
    """Thin wrapper so callers import one module for spectrum arms."""
    return coverage_masks_for_arm(
        arm=arm,
        atm=atm,
        option_labels=option_labels,
        equity_labels=equity_labels,
        spot=spot,
    )
