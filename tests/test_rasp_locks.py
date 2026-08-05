"""WP-P1: RASP shared locks and config-time refusals (table-driven)."""
from __future__ import annotations

import numpy as np
import pytest

from mascotrl.policy.rasp_locks import (
    MESSAGE_IDS,
    apply_rasp_defaults,
    assert_mask_honesty,
    assert_rasp_locks,
)
from mascotrl.spectrum.registry import validate_cfg


def _base(**overrides: object) -> dict:
    cfg: dict = {
        "train_world": "historical",
        "architecture": "mlp",
        "objective": "mean_std_cao",
        "algo": "ppo",
        "policy_mode": "shared",
        "projection_mode": "hard",
        "turnover_limit": 0.15,
        "weight_head": "softmax",
        "scr_mix": "off",
    }
    cfg.update(overrides)
    return cfg


@pytest.mark.parametrize(
    "overrides,message_id",
    [
        ({"weight_head": "dirichlet_tilt", "algo": "dqn", "objective": "differential_sharpe"}, "dirichlet_refuses_dqn"),
        ({"weight_head": "dirichlet_tilt", "algo": "happo"}, "dirichlet_refuses_happo"),
        ({"weight_head": "dirichlet_mean", "algo": "dqn", "objective": "differential_sharpe"}, "dirichlet_refuses_dqn"),
        (
            {"scr_mix": "full", "algo": "sac", "objective": "differential_sharpe"},
            "scr_full_requires_ppo_historical",
        ),
        (
            {"scr_mix": "full", "algo": "ppo", "train_world": "rbergomi"},
            "scr_full_requires_ppo_historical",
        ),
        (
            {"scr_mix": "full", "algo": "mcpg", "objective": "differential_sharpe"},
            "scr_full_requires_ppo_historical",
        ),
        (
            {"projection_mode": "soft", "turnover_limit": 0.15},
            "turnover_requires_hard_projection",
        ),
        (
            {"use_equity_feature_cube": True, "train_world": "heston"},
            "feature_cube_requires_historical",
        ),
    ],
)
def test_rasp_lock_refusals(overrides: dict, message_id: str) -> None:
    cfg = _base(**overrides)
    with pytest.raises(ValueError, match=message_id):
        assert_rasp_locks(cfg)
    assert message_id in MESSAGE_IDS


def test_legal_reference_cell_passes_rasp_locks() -> None:
    cfg = _base(weight_head="dirichlet_tilt", scr_mix="full")
    assert_rasp_locks(cfg)  # does not raise
    resolved = validate_cfg(cfg)
    assert resolved["algo"] == "ppo"
    assert resolved["train_world"] == "historical"


def test_validate_cfg_wires_rasp_locks() -> None:
    cfg = _base(weight_head="dirichlet_tilt", algo="dqn", objective="differential_sharpe")
    with pytest.raises(ValueError, match="illegal for algo=|dirichlet_refuses_dqn"):
        validate_cfg(cfg)


def test_legacy_soft_ofat_without_turnover_still_validates() -> None:
    """Existing OFAT cells use soft projection but omit turnover_limit."""
    cfg = {
        "train_world": "historical",
        "architecture": "mamba",
        "objective": "mean_std_cao",
        "algo": "ppo",
        "policy_mode": "shared",
        "projection_mode": "soft",
    }
    resolved = validate_cfg(cfg)
    assert resolved["architecture"] == "mamba"


def test_apply_rasp_defaults_auto_enables_feature_cube() -> None:
    cfg = _base(architecture="mamba")
    del cfg["weight_head"]
    out = apply_rasp_defaults(cfg)
    assert out["use_equity_feature_cube"] is True
    assert out["cube_auto_enabled"] is True


def test_mask_honesty_refuses_all_true_when_availability_exists() -> None:
    mask = np.ones((10, 5), dtype=bool)
    with pytest.raises(ValueError, match="mask_all_true_with_availability"):
        assert_mask_honesty(mask, availability_exists=True)


def test_mask_honesty_allows_all_true_without_availability() -> None:
    mask = np.ones((10, 5), dtype=bool)
    assert_mask_honesty(mask, availability_exists=False)


def test_mask_honesty_allows_partial_mask() -> None:
    mask = np.ones((10, 5), dtype=bool)
    mask[:, 0] = False
    assert_mask_honesty(mask, availability_exists=True)
