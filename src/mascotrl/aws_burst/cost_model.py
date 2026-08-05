"""Affordable-frontier cost governor for AWS Batch Spot waves (AWS-7)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mascotrl.aws_burst.profiles import BUDGET_USD, CREDIT_USD, SPEND_CAP_FRAC


@dataclass(frozen=True)
class CostEstimate:
    vcpus: int
    hours_per_cell: float
    usd_per_vcpu_hour: float
    n_cells: int

    @property
    def usd_total(self) -> float:
        return float(self.vcpus) * float(self.hours_per_cell) * float(
            self.usd_per_vcpu_hour
        ) * float(self.n_cells)


def affordable_frontier(
    *,
    n_cells: int,
    hours_per_cell_by_vcpu: dict[int, float],
    usd_per_vcpu_hour: float,
    budget_usd: float | None = None,
    credit_usd: float | None = None,
    spend_cap_frac: float | None = None,
) -> dict[str, Any]:
    """Pick the cost-minimising vCPU count that finishes under the spend cap."""
    bud = float(BUDGET_USD if budget_usd is None else budget_usd)
    cred = float(CREDIT_USD if credit_usd is None else credit_usd)
    frac = float(SPEND_CAP_FRAC if spend_cap_frac is None else spend_cap_frac)
    cap = min(bud, cred * frac)
    candidates: list[CostEstimate] = []
    for v, h in sorted(hours_per_cell_by_vcpu.items()):
        est = CostEstimate(
            vcpus=int(v),
            hours_per_cell=float(h),
            usd_per_vcpu_hour=float(usd_per_vcpu_hour),
            n_cells=int(n_cells),
        )
        if est.usd_total <= cap:
            candidates.append(est)
    if not candidates:
        return {
            "ok": False,
            "reason": "no_affordable_vcpu",
            "cap_usd": cap,
            "candidates": [],
        }
    best = min(candidates, key=lambda e: e.usd_total)
    return {
        "ok": True,
        "cap_usd": cap,
        "chosen_vcpus": best.vcpus,
        "usd_total": best.usd_total,
        "hours_per_cell": best.hours_per_cell,
        "candidates": [
            {
                "vcpus": c.vcpus,
                "usd_total": c.usd_total,
                "hours_per_cell": c.hours_per_cell,
            }
            for c in candidates
        ],
    }


def refuse_submit_if_unsafe(
    *,
    budget_action_armed: bool,
    projected_usd: float,
    credit_usd: float | None = None,
    spend_cap_frac: float | None = None,
) -> None:
    if not budget_action_armed:
        raise ValueError(
            "budget_action_not_armed: refuse submit without a live Budget Action "
            f"deny policy at 95% of the ${BUDGET_USD:.0f} budget"
        )
    cred = float(CREDIT_USD if credit_usd is None else credit_usd)
    frac = float(SPEND_CAP_FRAC if spend_cap_frac is None else spend_cap_frac)
    cap = cred * frac
    if float(projected_usd) > cap:
        raise ValueError(
            f"spend_cap_exceeded: projected_usd={projected_usd:.2f} > "
            f"{frac:.0%} of credit ({cap:.2f})"
        )
