"""Tests for interpretability engine."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.reporting.behavior_explain import explain_behaviour
from src.reporting.interpretability import (
    build_interpretability_artifact,
    channel_group_attribution,
    distill_policy_tree,
    prose_safe,
)


def _linear_policy(weights_from_obs: np.ndarray) -> callable:
    """Policy that tilts asset 0 by obs channel 0."""

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        x = float(np.asarray(obs).reshape(-1)[0])
        K = weights_from_obs.shape[0]
        w = np.full(K, 1.0 / K)
        w[0] = np.clip(0.5 + 0.2 * x, 0.05, 0.95)
        w[1:] = (1.0 - w[0]) / max(K - 1, 1)
        return w

    return policy_fn


def test_attribution_deterministic_given_seed() -> None:
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((20, 4))
    K = 4
    policy_fn = _linear_policy(np.zeros(K))
    groups = {"signal": [0], "noise": [1, 2, 3]}
    sleeve = np.eye(K, 7)
    a = channel_group_attribution(
        policy_fn=policy_fn,
        obs_matrix=obs,
        channel_groups=groups,
        sleeve_matrix=sleeve,
        n_shuffles=50,
        seed=42,
    )
    b = channel_group_attribution(
        policy_fn=policy_fn,
        obs_matrix=obs,
        channel_groups=groups,
        sleeve_matrix=sleeve,
        n_shuffles=50,
        seed=42,
    )
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_attribution_null_band_materiality_flag() -> None:
    rng = np.random.default_rng(1)
    T, K = 30, 3
    obs = rng.standard_normal((T, 4))
    obs[:, 0] = rng.standard_normal(T)

    def policy_fn(o: np.ndarray) -> np.ndarray:
        x = float(np.asarray(o).reshape(-1)[0])
        w = np.array([0.6, 0.25, 0.15])
        w[0] = np.clip(0.5 + 0.3 * x, 0.1, 0.9)
        w[1:] = (1.0 - w[0]) / 2.0
        return w

    groups = {"planted": [0], "decoy": [1, 2, 3]}
    result = channel_group_attribution(
        policy_fn=policy_fn,
        obs_matrix=obs,
        channel_groups=groups,
        sleeve_matrix=np.eye(K, 7),
        n_shuffles=100,
        seed=7,
    )
    assert result["groups"]["planted"]["l1_delta"] >= result["groups"]["decoy"]["l1_delta"]


def test_distillation_fidelity_gate() -> None:
    pytest.importorskip("sklearn")
    T, K = 40, 4
    sleeve = np.eye(K, 7)
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((T, K * 2))
    weights = np.full((T, K), 0.25)
    weights[:, 0] = 0.25 + 0.15 * obs[:, 0]
    weights = weights / weights.sum(axis=1, keepdims=True)

    good = distill_policy_tree(
        obs=obs,
        weights=weights,
        sleeve_matrix=sleeve,
        feature_names=["f0", "f1"],
        seed=0,
    )
    random_w = rng.random((T, K))
    random_w = random_w / random_w.sum(axis=1, keepdims=True)
    bad = distill_policy_tree(
        obs=rng.standard_normal((T, 8)),
        weights=random_w,
        sleeve_matrix=sleeve,
        feature_names=["f0"],
        seed=0,
    )
    assert good["distillable"] or np.isfinite(good["r2_oos"])
    assert not bad["distillable"] or bad["r2_oos"] < 0.5


def test_distillation_weight_only_path() -> None:
    pytest.importorskip("sklearn")
    T, K = 40, 5
    sleeve = np.eye(K, 7)
    w = np.full((T, K), 1.0 / K)
    w[:, 0] = np.linspace(0.15, 0.35, T)
    w = w / w.sum(axis=1, keepdims=True)
    result = distill_policy_tree(
        obs=None,
        weights=w,
        sleeve_matrix=sleeve,
        feature_names=[],
        seed=0,
    )
    assert "r2_oos" in result
    assert "distillable" in result


def test_mechanism_cards_no_causal_language() -> None:
    behaviour = {"turnover_mean": 0.1, "tilt_defensive": 0.05}
    by_regime = {
        "calm": {"tilt_defensive": 0.02, "tilt_lottery": 0.01, "turnover_mean": 0.08},
        "crisis": {"tilt_defensive": 0.12, "tilt_lottery": -0.01, "turnover_mean": 0.15},
    }
    macro = {
        "defensive": {"vix_z": {"coef": 0.05, "se": 0.01, "tstat": 2.0}},
    }
    explained = explain_behaviour(
        {"objective": "cvar_ru", "algo": "ppo", "architecture": "mlp"},
        behaviour,
        macro_sens=macro,
        behaviour_by_regime=by_regime,
    )
    for exp in explained["explanations"]:
        note = str(exp.get("note", ""))
        assert prose_safe(note), f"banned causal language in: {note}"


def test_artifact_honesty_locks() -> None:
    art = build_interpretability_artifact(
        cell_id="test_cell",
        attribution={"groups": {}, "top_groups": []},
        distillation={"distillable": False, "r2_oos": 0.0},
        mechanism_cards=[],
    )
    assert art["feeds_capital_gates"] is False
    assert art["interpretation_only"] is True
    assert art["schema_version"] == 1


def test_prose_safe_banned_words() -> None:
    assert prose_safe("co-movement with VIX z-score")
    assert not prose_safe("This causes higher turnover")
    assert not prose_safe("because of defensive tilt")


def test_interpretability_json_serializable() -> None:
    art = build_interpretability_artifact(
        cell_id="x",
        attribution={"top_groups": []},
        distillation={"r2_oos": 0.3, "distillable": False},
        mechanism_cards=[{"mechanism": "regime_shift_response", "verdict": "inconclusive"}],
    )
    json.dumps(art)
