"""RASP shared locks: config-time refusals and mask honesty.

Message ids are stable and greppable. See plan Part A.2 / A.5.
"""
from __future__ import annotations

from typing import Any, Mapping, MutableMapping

import numpy as np


MESSAGE_IDS: frozenset[str] = frozenset(
    {
        "dirichlet_refuses_dqn",
        "dirichlet_refuses_happo",
        "scr_full_requires_ppo_historical",
        "turnover_requires_hard_projection",
        "feature_cube_requires_historical",
        "mask_all_true_with_availability",
    }
)


def _weight_head(cfg: Mapping[str, Any]) -> str:
    return str(cfg.get("weight_head") or "softmax").lower().strip()


def _algo(cfg: Mapping[str, Any]) -> str:
    raw = cfg.get("algo") or cfg.get("policy_algo") or "ppo"
    return str(raw).lower().strip()


def _train_world(cfg: Mapping[str, Any]) -> str:
    raw = cfg.get("train_world") or cfg.get("train_distribution") or "historical"
    return str(raw).lower().strip()


def _is_dirichlet_head(head: str) -> bool:
    return head.startswith("dirichlet")


def assert_rasp_locks(cfg: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` with a stable ``message_id:`` prefix on violation."""
    head = _weight_head(cfg)
    algo = _algo(cfg)
    world = _train_world(cfg)
    scr = str(cfg.get("scr_mix") or "off").lower().strip()
    proj = str(cfg.get("projection_mode") or "soft").lower().strip()

    if _is_dirichlet_head(head) and algo == "dqn":
        raise ValueError(
            "dirichlet_refuses_dqn: weight_head starting with 'dirichlet' "
            "is incompatible with algo='dqn' (discrete foil only)"
        )
    if _is_dirichlet_head(head) and algo == "happo":
        raise ValueError(
            "dirichlet_refuses_happo: Dirichlet weight_head is refused for "
            "algo='happo' (use a separate HAPPO research flag, never the head enum)"
        )
    if scr == "full" and not (algo == "ppo" and world == "historical"):
        raise ValueError(
            "scr_full_requires_ppo_historical: scr_mix='full' requires "
            "algo='ppo' and train_world='historical'"
        )

    turnover_raw = cfg.get("turnover_limit")
    if turnover_raw is not None:
        try:
            tau = float(turnover_raw)
        except (TypeError, ValueError):
            tau = float("nan")
        if np.isfinite(tau) and proj != "hard":
            raise ValueError(
                "turnover_requires_hard_projection: finite turnover_limit "
                f"requires projection_mode='hard' (got {proj!r})"
            )

    if bool(cfg.get("use_equity_feature_cube", False)) and world != "historical":
        raise ValueError(
            "feature_cube_requires_historical: use_equity_feature_cube=true "
            f"requires train_world='historical' (got {world!r})"
        )


def apply_rasp_defaults(cfg: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Auto-enable feature cube for non-mlp historical bodies; stamp honesty."""
    arch = str(cfg.get("architecture") or cfg.get("temporal_backend") or "mlp").lower()
    world = _train_world(cfg)
    if arch != "mlp" and world == "historical" and not bool(cfg.get("use_equity_feature_cube")):
        cfg["use_equity_feature_cube"] = True
        cfg["cube_auto_enabled"] = True
    return cfg


def assert_mask_honesty(
    mask: np.ndarray,
    *,
    availability_exists: bool,
) -> None:
    """Refuse all-True dyn masks when lake availability data exists (L4)."""
    arr = np.asarray(mask)
    if availability_exists and arr.size > 0 and bool(np.all(arr)):
        raise ValueError(
            "mask_all_true_with_availability: slot_valid_mask is all-True "
            "while availability data exists; refuse mask theater"
        )
