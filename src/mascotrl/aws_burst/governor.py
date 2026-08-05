"""Per-wave cost projection and submit gates."""
from __future__ import annotations

import logging
from typing import Any

from src.aws_burst.budget_action import DEFAULT_BUDGET_NAME, read_actual_spend, spend_headroom
from src.aws_burst.cost_model import refuse_submit_if_unsafe
from pathlib import Path

from src.aws_burst.profiles import (
    BURST_PROFILES,
    CREDIT_USD,
    SPEND_CAP_FRAC,
    credit_usd_for_profile,
)

log = logging.getLogger(__name__)


def projected_wave_cost(
    *,
    n_cells: int,
    hours_per_cell: float,
    usd_per_vcpu_hour: float,
    vcpus: int = 1,
) -> float:
    return float(n_cells) * float(hours_per_cell) * float(usd_per_vcpu_hour) * float(vcpus)


def check_submit_allowed(
    *,
    armed_profiles: list[dict[str, str]],
    projected_usd: float,
    clients: list[Any] | None = None,
    offline: bool = False,
    budget_name: str = DEFAULT_BUDGET_NAME,
    allow_partial_profiles: bool = False,
    root: Path | None = None,
) -> None:
    # Full fleet waves require all 3 armed accounts. Smoke / staged rollout may
    # pass allow_partial_profiles=True with an explicit --profiles filter.
    if not allow_partial_profiles and len(armed_profiles) != len(BURST_PROFILES):
        raise ValueError(
            f"incomplete_armed_profiles: need all {len(BURST_PROFILES)}, "
            f"got {len(armed_profiles)}"
        )
    if allow_partial_profiles and not armed_profiles:
        raise ValueError("incomplete_armed_profiles: need at least 1, got 0")
    if root is not None and armed_profiles:
        credit = min(
            credit_usd_for_profile(root, p["profile"]) for p in armed_profiles
        )
    else:
        credit = CREDIT_USD
    refuse_submit_if_unsafe(
        budget_action_armed=bool(armed_profiles),
        projected_usd=projected_usd,
        credit_usd=credit,
        spend_cap_frac=SPEND_CAP_FRAC,
    )
    if offline or not clients:
        log.info("budget_armed_files_present_offline_or_no_clients")
        return

    for client in clients:
        actual = read_actual_spend(client, budget_name=budget_name)
        headroom = spend_headroom(actual_usd=actual, projected_usd=projected_usd)
        profile = getattr(client, "profile", "?")
        log.info(
            "budget_live_spend profile=%s actual_usd=%.4f projected_usd=%.4f headroom=%.4f",
            profile,
            actual,
            projected_usd,
            headroom,
        )
        prof_credit = (
            credit_usd_for_profile(root, profile)
            if root is not None
            else CREDIT_USD
        )
        if headroom < 0:
            raise ValueError(
                f"spend_cap_exceeded_live: profile={profile} "
                f"actual={actual:.2f} + projected={projected_usd:.2f} "
                f"> {SPEND_CAP_FRAC:.0%} of credit "
                f"({prof_credit * SPEND_CAP_FRAC:.2f})"
            )
