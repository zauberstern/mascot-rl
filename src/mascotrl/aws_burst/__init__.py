"""AWS Frankfurt burst helpers (isolated from deploy/aws_free_tier)."""
from __future__ import annotations

from mascotrl.aws_burst.cost_model import affordable_frontier, refuse_submit_if_unsafe
from mascotrl.aws_burst.profiles import BURST_PROFILES, REGION

__all__ = [
    "BURST_PROFILES",
    "REGION",
    "affordable_frontier",
    "refuse_submit_if_unsafe",
]
