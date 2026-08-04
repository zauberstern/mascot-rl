"""Protocol tiers for the spectrum study (WP-S5).

CPCV geometry is identical across confirmatory/parity/screening; only seeds and
train budget differ. Narrative uses a wider CPCV geometry by design.
Tiers are budget/geometry only; the this repo makes no capital-allocation claims.
"""
from __future__ import annotations

from typing import Any

PROTOCOL_TIERS = ("confirmatory", "parity", "screening", "narrative")

_TIER_SPECS: dict[str, dict[str, Any]] = {
    "confirmatory": {
        "seeds": list(range(10)),
        "train_env_steps": 100_000,
        "cpcv_n_splits": 6,
        "cpcv_n_test_groups": 2,
        "cpcv_purge_days": 21,
        "cpcv_embargo_days": 21,
    },
    # Cherry-pick Phase 5: pre-registered behaviour narrative.
    # Wider CPCV geometry (8/3) than confirmatory/parity/screening (6/2).
    "narrative": {
        "seeds": list(range(10)),
        "train_env_steps": 100_000,
        "cpcv_n_splits": 8,
        "cpcv_n_test_groups": 3,
        "cpcv_purge_days": 21,
        "cpcv_embargo_days": 21,
    },
    "parity": {
        "seeds": [0, 1, 2],
        "train_env_steps": 100_000,
        "cpcv_n_splits": 6,
        "cpcv_n_test_groups": 2,
        "cpcv_purge_days": 21,
        "cpcv_embargo_days": 21,
    },
    "screening": {
        "seeds": [0],
        "train_env_steps": 25_000,
        "cpcv_n_splits": 6,
        "cpcv_n_test_groups": 2,
        "cpcv_purge_days": 21,
        "cpcv_embargo_days": 21,
    },
}


def resolve_protocol_tier(name: str) -> dict[str, Any]:
    key = str(name or "").lower().strip()
    if key not in _TIER_SPECS:
        raise ValueError(
            f"unknown protocol_tier={name!r}; allowed={list(PROTOCOL_TIERS)}"
        )
    out = dict(_TIER_SPECS[key])
    out["protocol_tier"] = key
    out["n_seeds"] = len(out["seeds"])
    return out


def apply_protocol_tier(cfg: dict[str, Any], tier: str) -> dict[str, Any]:
    """Stamp tier fields onto a cell cfg (mutates and returns)."""
    spec = resolve_protocol_tier(tier)
    cfg["protocol_tier"] = spec["protocol_tier"]
    cfg["seeds"] = list(spec["seeds"])
    cfg["train_env_steps"] = int(spec["train_env_steps"])
    cfg.pop("capital_eligible", None)
    for k in (
        "cpcv_n_splits",
        "cpcv_n_test_groups",
        "cpcv_purge_days",
        "cpcv_embargo_days",
    ):
        cfg[k] = int(spec[k])
    return cfg
