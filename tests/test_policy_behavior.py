"""W7 / Part E policy behavior harness (schema v2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mascotrl.reporting.policy_behavior import (
    ARCHETYPE_IDS,
    ARCHETYPE_SCORE_WEIGHTS,
    build_policy_behavior,
    plot_archetype_figures,
    weight_concentration,
    write_policy_behavior,
)


def test_weight_concentration_ew():
    k = 10
    w = np.full((5, k), 1.0 / k)
    out = weight_concentration(w)
    assert abs(out["hhi_mean"] - 1.0 / k) < 1e-9
    assert out["l1_vs_ew_mean"] < 1e-9


def test_archetype_ids_are_measured_not_algo_cards():
    assert len(ARCHETYPE_IDS) == 5
    assert "trend_follower" in ARCHETYPE_SCORE_WEIGHTS
    assert "carry_harvester" not in ARCHETYPE_SCORE_WEIGHTS
    assert "ppo" not in ARCHETYPE_SCORE_WEIGHTS


def test_build_and_write_policy_behavior_v2(tmp_path: Path):
    w = np.full((12, 4), 0.25)
    s = np.eye(4, 7)
    r = np.random.default_rng(1).normal(0, 0.01, size=w.shape)
    payload = build_policy_behavior(
        algo="ppo",
        architecture="mlp",
        objective="mean_std_cao",
        policy_mode="balanced",
        weights=w,
        asset_returns=r,
        sleeve_matrix=s,
        n_null_shuffles=20,
        extras={
            "seed_sharpes": [1.0, 1.01, 0.99],
            "cmdp_slack_series": [0.0, 0.01, 0.0],
        },
    )
    assert payload["schema_version"] == 2
    assert payload["interpretation_only"] is True
    assert payload["feeds_capital_gates"] is False
    assert payload["archetype_primary"] in set(ARCHETYPE_IDS) | {"mixed"}
    assert set(payload["archetype_scores"]) == set(ARCHETYPE_IDS)
    path = write_policy_behavior(tmp_path / "policy_behavior.json", payload)
    assert path.is_file()
    figs = plot_archetype_figures(payload, tmp_path / "figs")
    assert len(figs) >= 2
    assert any(Path(f).name == "archetype_cmdp_slack.png" for f in figs)
    for f in figs:
        assert Path(f).is_file()
