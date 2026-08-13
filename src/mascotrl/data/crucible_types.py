"""CRUCIBLE constants and datatypes."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

SLEEVE_IDS: tuple[str, ...] = (
    "trend",
    "reversal",
    "carry",
    "defensive",
    "lottery",
    "illiquid",
    "core",
)

SLEEVE_QUOTAS: dict[str, int] = {
    "trend": 18,
    "reversal": 18,
    "carry": 18,
    "defensive": 14,
    "lottery": 12,
    "illiquid": 10,
    "core": 10,
}

SLEEVE_FILL_ORDER: tuple[str, ...] = (
    "illiquid",
    "lottery",
    "defensive",
    "carry",
    "reversal",
    "trend",
    "core",
)

_SLEEVE_DEFS_PAYLOAD = {
    "trend": "12m_resid_skip_21d",
    "reversal": "neg_21d_resid",
    "carry": "iv30_minus_hv20",
    "defensive": "neg_resid_vol_plus_neg_beta",
    "lottery": "resid_idio_vol_plus_abs_skew",
    "illiquid": "neg_adv_rank",
    "core": "overall_rank_buffer",
}


def sleeve_defs_hash() -> str:
    return hashlib.sha256(
        json.dumps(_SLEEVE_DEFS_PAYLOAD, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass
class CrucibleSpec:
    k: int = 100
    max_pool: int = 511
    lookback_days: int = 252
    reselect_every_days: int = 63
    reselect_churn_cap: float = 0.25
    adv_strata: tuple[tuple[float, float, float], ...] = (
        (70.0, 100.0, 0.40),
        (40.0, 70.0, 0.40),
        (20.0, 40.0, 0.20),
    )
    adv_participation_floor: float = 0.10
    amihud_drop_pct: float = 95.0
    amihud_drop_pct_crisis: float = 90.0
    n_communities: int = 20
    max_per_community: int = 3
    quotas: dict[str, int] = field(default_factory=lambda: dict(SLEEVE_QUOTAS))
    lottery_resid_var_share_cap: float = 0.20
    g1_l1_floor: float = 0.08
    g1_entropy_gap_floor: float = 0.60
    g2_tc_floor: float = 0.35
    g3_sharpe_floor: float = 0.10
    g3_participation_ladder: tuple[float, ...] = (0.01, 0.05, 0.10)
    max_repair_passes: int = 5

    def assert_k_feasible(self) -> None:
        """Fail closed when community packing cannot physically fill ``k``.

        ``n_communities * max_per_community`` is a hard upper bound on the
        selected set. A K=100 confirmatory with (20, 3) silently under-fills
        (~57 names) and then crashes inside G1 with a cryptic projector
        size mismatch; catch the config error here instead.
        """
        cap = int(self.n_communities) * int(self.max_per_community)
        if cap < int(self.k):
            raise ValueError(
                f"CRUCIBLE community capacity {self.n_communities}×"
                f"{self.max_per_community}={cap} < k={self.k}; raise "
                f"n_communities and/or max_per_community before launch"
            )

@dataclass(frozen=True)
class CrucibleResult:
    secids: list[int]
    sleeve_membership: dict[str, list[int]]
    sleeve_primary: dict[int, str]
    sleeve_matrix: np.ndarray
    community_of: dict[int, int]
    partition_scores: list[int]
    diagnostics: dict
    fingerprint: str


class CrucibleGateFailure(RuntimeError):
    """Raised when G1/G2/G3 still fail after repair."""

    def __init__(self, message: str, diagnostics: dict | None = None):
        self.diagnostics = diagnostics or {}
        g1 = self.diagnostics.get("g1_pass")
        g2 = self.diagnostics.get("g2_pass")
        g3 = self.diagnostics.get("g3_pass")
        detail = (
            f"{message} "
            f"(g1_pass={g1}, g2_pass={g2}, g3_pass={g3}, "
            f"g1_gap={self.diagnostics.get('g1_entropy_gap')}, "
            f"g1_gap_floor_eff={self.diagnostics.get('g1_entropy_gap_floor_effective')}, "
            f"g2_tc={self.diagnostics.get('g2_tc_post_projection')}, "
            f"repair_passes_used={self.diagnostics.get('repair_passes_used')})"
        )
        super().__init__(detail)

