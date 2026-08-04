"""Archetype-driven codenames with frozen animal table (C5)."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.assign_behavior_codenames import (
    ARCHETYPE_TO_ANIMAL,
    cluster_behaviours,
)


def _write_beh(path: Path, arch: str, scores: dict[str, float], behaviour: dict) -> None:
    payload = {
        "archetype_primary": arch,
        "archetype_scores": scores,
        "behaviour": behaviour,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_codename_maps_archetype_to_animal(tmp_path: Path) -> None:
    b1 = tmp_path / "cell_a_policy_behavior.json"
    b2 = tmp_path / "cell_b_policy_behavior.json"
    beh = {
        "turnover_mean": 0.2,
        "tilt_trend": 0.5,
        "hhi_mean": 0.1,
        "tilt_reversal": 0.05,
        "tilt_carry": 0.1,
        "tilt_defensive": 0.05,
        "tilt_lottery": 0.02,
        "tilt_illiquid": 0.02,
        "tilt_autocorr_lag21": 0.3,
        "rotation_rate": 0.1,
        "holding_period_days": 20.0,
        "l1_vs_ew_mean": 0.1,
        "max_weight_mean": 0.15,
        "downside_capture": 0.8,
        "upside_capture": 1.1,
        "return_skew": 0.0,
        "max_drawdown": 0.1,
        "cvar_05": 0.05,
        "turnover_cap_binding_frac": 0.0,
        "action_entropy_mean": 1.0,
        "weight_autocorr_lag1": 0.5,
        "tilt_core": 0.2,
        "across_regime_tilt_variance": 0.01,
        "within_regime_tilt_variance": 0.01,
    }
    scores_tf = {k: 0.1 for k in ARCHETYPE_TO_ANIMAL if k != "mixed"}
    scores_tf["trend_follower"] = 2.0
    _write_beh(b1, "trend_follower", scores_tf, beh)
    scores_rm = dict(scores_tf)
    scores_rm["trend_follower"] = 0.1
    scores_rm["risk_manager"] = 2.0
    beh_rm = dict(beh)
    beh_rm["tilt_defensive"] = 0.6
    _write_beh(b2, "risk_manager", scores_rm, beh_rm)
    out = cluster_behaviours([b1, b2], k=2)
    assert out["ok"] is True
    cells = {c["cell_id"]: c for c in out["cells"]}
    assert cells["cell_a_policy_behavior"]["animal_mascot"] == ARCHETYPE_TO_ANIMAL["trend_follower"]
    assert cells["cell_b_policy_behavior"]["animal_mascot"] == ARCHETYPE_TO_ANIMAL["risk_manager"]
    assert "trend" in cells["cell_a_policy_behavior"]["why"].lower()
