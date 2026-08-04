"""Universe capacity probe for the spectrum K-axis (WP-S2)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class CapacityProbeResult:
    pool_size: int
    requested_k: tuple[int, ...]
    feasible_k: tuple[int, ...]
    refused_k: tuple[int, ...]
    k_axis: tuple[int, ...]
    k_max: int
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe_universe_capacity(
    pool_size: int,
    *,
    requested: Sequence[int] = (100, 200, 400),
) -> CapacityProbeResult:
    """Derive ``{100, 200, K_max}`` and ledger any infeasible request (e.g. 400)."""
    n = int(pool_size)
    if n <= 0:
        raise ValueError(f"pool_size must be positive, got {pool_size}")
    req = tuple(int(k) for k in requested)
    feasible = tuple(k for k in req if k <= n)
    refused = tuple(k for k in req if k > n)
    # Always include the largest feasible breadth as K_max when 400 is refused.
    k_max = max(feasible) if feasible else min(n, max(req))
    if refused and k_max not in feasible:
        # Promote pool size itself as K_max when no requested K fits.
        k_max = n
    # Study axis: requested feasibles plus K_max if distinct and feasible.
    axis: list[int] = []
    for k in sorted(set(feasible) | ({k_max} if k_max <= n else set())):
        if k not in axis and k <= n:
            axis.append(k)
    if not axis:
        axis = [min(n, 100)]
        k_max = axis[0]
    note = ""
    if refused:
        note = (
            f"refused_k={list(refused)}: pool_size={n} insufficient; "
            f"using k_axis={axis} with k_max={k_max}"
        )
    return CapacityProbeResult(
        pool_size=n,
        requested_k=req,
        feasible_k=feasible,
        refused_k=refused,
        k_axis=tuple(axis),
        k_max=int(k_max),
        note=note,
    )
