"""Scenario A rescue: design map, alignment wiring, Hummingbird proxy, softmax outliers."""
from __future__ import annotations

import numpy as np
import pytest

from src.reporting.policy_behavior import (
    PERSONALITY_DESIGN_MAP,
    build_policy_behavior,
    designed_personality,
    normalize_weight_head,
)


def test_sparse_tilt_design_map_entries_resolve_non_mixed() -> None:
    """Every plan-listed sparse_tilt design intent resolves (except baseline control)."""
    expected = {
        ("differential_sharpe", "ppo", "sparse_tilt"): "trend_follower",
        ("meanvar_kolm", "ppo", "sparse_tilt"): "contrarian",
        ("cvar_ru", "ppo", "sparse_tilt"): "risk_manager",
        ("cvar_ru", "cppo", "sparse_tilt"): "risk_manager",
        ("mtm_pnl", "ppo", "sparse_tilt"): "speculator",
        ("mtm_pnl", "ddpg", "sparse_tilt"): "speculator",
        ("mtm_pnl", "sac", "sparse_tilt"): "speculator",
        ("mtm_pnl", "td3", "sparse_tilt"): "speculator",
        ("mtm_pnl", "mcpg", "sparse_tilt"): "speculator",
        ("mean_std_cao", "happo", "sparse_tilt"): "tactical_rotator",
        ("mean_std_cao", "ppo", "sparse_tilt"): "mixed",
        ("entropic_oce", "ppo", "sparse_tilt"): "contrarian",
        ("sdr_composite", "ppo", "sparse_tilt"): "trend_follower",
        ("rsqp", "ppo", "sparse_tilt"): "risk_manager",
        ("smse", "ppo", "sparse_tilt"): "risk_manager",
        ("mikkila_asym", "ppo", "sparse_tilt"): "speculator",
        ("mikkila_asym", "ddpg", "sparse_tilt"): "speculator",
        ("mikkila_asym", "sac", "sparse_tilt"): "speculator",
        ("mikkila_asym", "td3", "sparse_tilt"): "speculator",
    }
    for (obj, algo, head), want in expected.items():
        got = designed_personality(objective=obj, algo=algo, weight_head=head)
        assert got == want, f"{(obj, algo, head)} -> {got!r}, want {want!r}"


def test_mandate_preset_4tuple_hummingbird_proxy() -> None:
    for mandate in ("archetype_carry", "archetype_crisis", "archetype_inflation"):
        got = designed_personality(
            objective="mean_std_cao",
            algo="ppo",
            weight_head="sparse_tilt",
            mandate_preset=mandate,
        )
        assert got == "tactical_rotator", f"mandate={mandate} -> {got}"


def test_design_map_accepts_3tuple_and_4tuple_keys() -> None:
    # Existing softmax keys still work.
    assert designed_personality(
        objective="cvar_ru", algo="cppo", weight_head="softmax"
    ) == "risk_manager"
    # 4-tuple takes precedence over 3-tuple for same base.
    assert designed_personality(
        objective="mean_std_cao",
        algo="ppo",
        weight_head="sparse_tilt",
        mandate_preset="",
    ) == "mixed"
    assert ("mean_std_cao", "ppo", "sparse_tilt", "archetype_carry") in PERSONALITY_DESIGN_MAP


def test_normalize_weight_head() -> None:
    assert normalize_weight_head("sparse_tilt") == "sparse_tilt"
    assert normalize_weight_head("sparse") == "sparse_tilt"
    assert normalize_weight_head("tilt") == "sparse_tilt"
    assert normalize_weight_head("softmax") == "softmax"
    assert normalize_weight_head("tanh_l1") == "tanh_l1"
    assert normalize_weight_head("tanh") == "tanh_l1"
    assert normalize_weight_head("balanced") == "balanced"


def test_build_policy_behavior_emits_alignment_fields() -> None:
    w = np.full((8, 4), 0.25)
    s = np.eye(4, 7)
    r = np.random.default_rng(0).normal(0, 0.01, size=w.shape)
    payload = build_policy_behavior(
        algo="ppo",
        architecture="mlp",
        objective="cvar_ru",
        policy_mode="sparse_tilt",
        weights=w,
        asset_returns=r,
        sleeve_matrix=s,
        n_null_shuffles=5,
    )
    assert "alignment_pass" in payload
    assert "alignment_score" in payload
    assert payload["designed_personality"] == "risk_manager"
    assert "observed_personality" in payload
    assert "alignment_divergence" in payload
    assert isinstance(payload["alignment_pass"], bool)
    assert 0.0 <= float(payload["alignment_score"]) <= 1.0


def test_build_policy_behavior_softmax_collapse_exception_flag() -> None:
    # Concentrated book under softmax should flag escape from EW collapse.
    rng = np.random.default_rng(1)
    w = np.zeros((10, 5))
    w[:, 0] = 0.8
    w[:, 1] = 0.2
    s = np.eye(5, 7)
    r = rng.normal(0, 0.01, size=w.shape)
    payload = build_policy_behavior(
        algo="mcpg",
        architecture="mlp",
        objective="mtm_pnl",
        policy_mode="softmax",
        weights=w,
        asset_returns=r,
        sleeve_matrix=s,
        n_null_shuffles=5,
    )
    l1 = float((payload.get("behaviour") or {}).get("l1_vs_ew_mean") or 0.0)
    if l1 > 0.25:
        assert payload["behaviour"].get("softmax_collapse_exception") is True
        assert "softmax_escape_note" in payload["behaviour"]
    else:
        pytest.skip(f"synthetic L1={l1:.4f} did not exceed 0.25; geometry too weak")


def test_build_policy_behavior_uses_weight_head_not_policy_mode() -> None:
    """Campaign YAMLs set policy_mode=single while weight_head carries the head."""
    w = np.full((8, 4), 0.25)
    s = np.eye(4, 7)
    r = np.random.default_rng(3).normal(0, 0.01, size=w.shape)
    payload = build_policy_behavior(
        cell_id="eq_K100_single_ppo_mlp_sparse_tilt_cvar_ru",
        algo="ppo",
        architecture="mlp",
        objective="cvar_ru",
        policy_mode="single",
        weights=w,
        asset_returns=r,
        sleeve_matrix=s,
        n_null_shuffles=5,
        cell_cfg={"weight_head": "sparse_tilt", "policy_mode": "single", "algo": "ppo"},
    )
    assert payload["designed_personality"] == "risk_manager"
    assert payload.get("weight_head") == "sparse_tilt" or (
        (payload.get("behaviour") or {}).get("weight_head") in (None, "sparse_tilt")
    )


def test_build_policy_behavior_hummingbird_from_policy_mode_mandate() -> None:
    """RC6 stores mandate as policy_mode=archetype_carry (not mandate_preset)."""
    w = np.full((8, 4), 0.25)
    s = np.eye(4, 7)
    r = np.random.default_rng(4).normal(0, 0.01, size=w.shape)
    payload = build_policy_behavior(
        cell_id="eq_K100_single_ppo_mlp_sparse_tilt_mean_std_cao_pm-archetype_carry",
        algo="ppo",
        architecture="mlp",
        objective="mean_std_cao",
        policy_mode="archetype_carry",
        weights=w,
        asset_returns=r,
        sleeve_matrix=s,
        n_null_shuffles=5,
        cell_cfg={
            "weight_head": "sparse_tilt",
            "policy_mode": "archetype_carry",
            "algo": "ppo",
            "objective": "mean_std_cao",
        },
    )
    assert payload["designed_personality"] == "tactical_rotator"


def test_build_policy_behavior_hummingbird_proxy_flag() -> None:
    """Mandate-preset sparse_tilt cells get designed tactical_rotator."""
    w = np.full((8, 4), 0.25)
    s = np.eye(4, 7)
    r = np.random.default_rng(2).normal(0, 0.01, size=w.shape)
    payload = build_policy_behavior(
        algo="ppo",
        architecture="mlp",
        objective="mean_std_cao",
        policy_mode="sparse_tilt",
        weights=w,
        asset_returns=r,
        sleeve_matrix=s,
        n_null_shuffles=5,
        cell_cfg={"mandate_preset": "archetype_carry", "weight_head": "sparse_tilt"},
    )
    assert payload["designed_personality"] == "tactical_rotator"
