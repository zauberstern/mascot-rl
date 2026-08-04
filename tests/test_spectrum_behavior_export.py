"""Behaviour export must receive hoisted OOS weights (archetype narrative)."""
from __future__ import annotations

import pytest
from tests.conftest import FLOAT_TOL

import json
import math
from pathlib import Path

import numpy as np
import yaml

from scripts.run_spectrum_campaign import (
    _hoist_runner_weights,
    resolve_spectrum_budget,
    run_cell,
)
from src.reporting.policy_behavior import build_policy_behavior, write_policy_behavior


def test_hoist_runner_weights_from_path0() -> None:
    w = np.array([[0.5, 0.5], [0.25, 0.75], [0.1, 0.9]], dtype=np.float64)
    runner = {
        "paths": {"0": {"weights": w.tolist(), "turnover": [0.1, 0.2, 0.15]}},
        "n_seeds": 1,
        "policy_diagnostics": {"hhi_mean": 0.5},
    }
    hoisted = _hoist_runner_weights(runner)
    assert "weights" in hoisted
    assert np.asarray(hoisted["weights"]).shape == (3, 2)
    assert hoisted["training_diagnostics"]["hhi_mean"] == pytest.approx(0.5, **FLOAT_TOL)
    assert hoisted.get("turnovers") == [0.1, 0.2, 0.15]


def test_hoist_missing_weights_returns_empty() -> None:
    assert _hoist_runner_weights(None) == {}
    assert _hoist_runner_weights({"paths": {}}) == {}


def test_narrative_budget_opts_out_of_dispatch_smoke() -> None:
    smoke = resolve_spectrum_budget(
        {
            "claim_tier": "dispatch_only",
            "train_episodes": 50,
            "spectrum_happo_horizon": 64,
        }
    )
    assert smoke["dispatch_only"] is True
    assert smoke["n_episodes"] <= 2

    narrative = resolve_spectrum_budget(
        {
            "claim_tier": "dispatch_only",
            "protocol_tier": "narrative",
            "train_episodes": 50,
            "spectrum_happo_horizon": 64,
            "happo_full_budget": True,
        }
    )
    assert narrative["dispatch_only"] is False
    assert narrative["n_episodes"] == 50
    assert narrative["horizon"] == 64
    assert narrative["claim_tier"] != "dispatch_only"


def test_policy_behavior_export_finite_measures(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    weights = rng.dirichlet(np.ones(6), size=40)
    beh = build_policy_behavior(
        cell_id="toy_cell_eq_ppo",
        arm="eq",
        algo="ppo",
        architecture="mlp",
        objective="mean_std_cao",
        weights=weights,
        turnovers=rng.uniform(0.01, 0.2, size=40).tolist(),
    )
    assert beh["feeds_capital_gates"] is False
    behaviour = beh["behaviour"]
    finite = [
        v
        for v in behaviour.values()
        if isinstance(v, (int, float)) and math.isfinite(float(v))
    ]
    assert len(finite) >= 10
    assert beh.get("archetype_primary")
    out = tmp_path / "toy_cell_eq_ppo_policy_behavior.json"
    write_policy_behavior(out, beh)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["feeds_capital_gates"] is False
    assert len(payload["behaviour"]) >= 10


def test_run_cell_hoists_weights_when_runner_nested(tmp_path: Path, monkeypatch) -> None:
    """run_cell must surface path-0 weights on the top-level artifact."""
    cfg_path = tmp_path / "eq_K100_single_ppo_mlp_softmax_mean_std_cao.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "algo": "ppo",
                "architecture": "mlp",
                "objective": "mean_std_cao",
                "train_world": "historical",
                "portfolio_arm": "eq",
                "n_assets": 4,
                "grid_kind": "cherrypick",
                "claim_label_stem": "stk_ret",
                "spectrum_cell_id": "eq_K100_single_ppo_mlp_softmax_mean_std_cao",
            }
        ),
        encoding="utf-8",
    )
    fake_weights = [[0.5, 0.5, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]]

    def _fake_research_arm(cfg, arm, *, allow_toy_panel=False, no_dry_run=False, **_kw):
        return (
            {
                "path_summary": {"sharpe_mean": 0.1},
                "policy": {"sharpe_mean": 0.1},
                "paths": {"0": {"weights": fake_weights, "turnover": [0.05, 0.06]}},
                "panel_source": "toy",
                "toy_panel": True,
                "real_reference_arm_present": True,
                "policy_diagnostics": {"hhi_mean": 0.4},
            },
            None,
        )

    import scripts.run_spectrum_campaign as camp

    monkeypatch.setattr(camp, "_run_research_arm", _fake_research_arm)
    art = run_cell(cfg_path, dry_run=False, allow_toy_panel=True)
    assert art.get("behaviour_export") == "ready"
    assert art.get("weights") == fake_weights
    assert art.get("training_diagnostics", {}).get("hhi_mean") == pytest.approx(0.4, **FLOAT_TOL)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    beh = build_policy_behavior(
        cell_id=str(art["spectrum_cell_id"]),
        arm=str(art["arm"]),
        cell_cfg=art,
        weights=np.asarray(art["weights"]),
        turnovers=art.get("turnovers"),
    )
    beh_path = out_dir / f"{art['spectrum_cell_id']}_policy_behavior.json"
    write_policy_behavior(beh_path, beh)
    payload = json.loads(beh_path.read_text(encoding="utf-8"))
    finite = [
        v
        for v in payload["behaviour"].values()
        if isinstance(v, (int, float)) and math.isfinite(float(v))
    ]
    assert len(finite) >= 10
    assert payload["feeds_capital_gates"] is False
    assert payload.get("archetype_primary")


def test_panel_zscore_produces_nonzero_archetype_scores() -> None:
    """When scoring across 15+ cells, z-scores must be non-zero (RC5)."""
    from src.reporting.policy_behavior import assign_archetype, score_archetypes

    rng = np.random.default_rng(42)
    rows = []
    for _ in range(15):
        rows.append(
            {
                "hhi_mean": float(rng.uniform(0.01, 0.5)),
                "tilt_trend": float(rng.normal(0, 0.1)),
                "tilt_carry": float(rng.normal(0, 0.1)),
                "tilt_defensive": float(rng.normal(0, 0.1)),
                "tilt_reversal": float(rng.normal(0, 0.1)),
                "tilt_lottery": float(rng.normal(0, 0.05)),
                "tilt_illiquid": float(rng.normal(0, 0.05)),
                "tilt_core": float(rng.normal(0, 0.05)),
                "turnover_mean": float(rng.uniform(0.01, 0.3)),
                "holding_period_days": float(rng.uniform(1, 60)),
                "n_eff_mean": float(rng.uniform(2, 50)),
                "max_weight_mean": float(rng.uniform(0.05, 0.4)),
                "l1_vs_ew_mean": float(rng.uniform(0.0, 0.8)),
                "downside_capture": float(rng.uniform(0.5, 1.5)),
                "upside_capture": float(rng.uniform(0.5, 1.5)),
                "return_skew": float(rng.normal(0, 0.5)),
                "max_drawdown": float(rng.uniform(-0.4, -0.05)),
                "cvar_05": float(rng.uniform(-0.05, -0.01)),
                "action_entropy_mean": float(rng.uniform(0.5, 4.0)),
                "weight_autocorr_lag1": float(rng.uniform(0.0, 0.9)),
                "tilt_autocorr_lag21": float(rng.uniform(-0.2, 0.8)),
                "rotation_rate": float(rng.uniform(0.0, 0.5)),
                "turnover_cap_binding_frac": float(rng.uniform(0.0, 0.5)),
            }
        )
    scores = score_archetypes(rows)
    any_nonzero = any(any(abs(v) > 1e-9 for v in s.values()) for s in scores)
    assert any_nonzero, "panel z-scoring must produce non-zero scores"
    decisions = [assign_archetype(s) for s in scores]
    non_mixed = [d for d in decisions if d["archetype_primary"] != "mixed"]
    assert len(non_mixed) >= 1, "at least one cell should have a non-mixed archetype"


def test_refresh_behavior_exports_panel_rescores(tmp_path: Path, monkeypatch) -> None:
    """refresh_behavior_exports must pass a multi-cell behaviour_panel (RC5)."""
    from scripts import run_spectrum_campaign as camp
    from src.reporting import policy_behavior as pb

    captured_panels: list[int] = []
    real = pb.build_policy_behavior

    def spy_build(*args, behaviour_panel=None, **kwargs):
        captured_panels.append(0 if behaviour_panel is None else len(list(behaviour_panel)))
        return real(*args, behaviour_panel=behaviour_panel, **kwargs)

    monkeypatch.setattr(pb, "build_policy_behavior", spy_build)
    # Avoid lake I/O in unit test.
    monkeypatch.setattr(
        camp,
        "_behaviour_context_for_cell",
        lambda art, cfg, runner: {
            "dates": [],
            "sleeve_matrix": np.eye(4, 7),
            "universe_fingerprint": "",
            "regimes": None,
            "vix_z": None,
            "hy_oas_z": None,
            "term_spread": None,
            "asset_returns": np.random.default_rng(0).normal(0, 0.01, size=(8, 4)),
            "macro_status": {"status": "test"},
            "turnover_cap": None,
            "secids": [],
            "eval_dates": [],
        },
    )

    for i, tilt in enumerate([0.5, 2.0]):
        cell = f"toy_cell_{i}"
        K = 4
        w = np.full((8, K), 1.0 / K)
        w[:, 0] = tilt / K
        w = w / w.sum(axis=1, keepdims=True)
        art = {
            "spectrum_cell_id": cell,
            "arm": "eq",
            "weights": w.tolist(),
            "turnovers": [0.05] * 8,
            "n_assets": K,
            "panel_returns": np.random.default_rng(i).normal(0, 0.01, size=(8, K)).tolist(),
            "policy_returns": np.random.default_rng(i).normal(0, 0.01, size=8).tolist(),
            "factors": np.random.default_rng(i).normal(0, 0.01, size=(8, 3)).tolist(),
            "factor_names": ["f0", "f1", "f2"],
        }
        (tmp_path / f"{cell}.json").write_text(json.dumps(art), encoding="utf-8")
        # Minimal YAML so weight_head/objective are available for the probe.
        (tmp_path / f"{cell}.yaml").write_text(
            "algo: ppo\narchitecture: mlp\nobjective: mean_std_cao\n"
            "weight_head: softmax\ntrain_world: historical\npolicy_mode: shared\n",
            encoding="utf-8",
        )

    summary = camp.refresh_behavior_exports(tmp_path, config_dir=tmp_path)
    assert len(summary["refreshed"]) == 2
    assert captured_panels, "build_policy_behavior never called"
    assert all(n >= 1 for n in captured_panels), (
        f"expected peer panel for each cell, got sizes={captured_panels}"
    )
    # Composition fields stamped on refreshed outputs.
    for i in range(2):
        beh = json.loads(
            (tmp_path / f"toy_cell_{i}_policy_behavior.json").read_text(encoding="utf-8")
        )
        assert "archetype_composition" in beh
        assert "archetype_confidence" in beh
        comp = beh["archetype_composition"]
        assert abs(sum(comp.values()) - 1.0) < 1e-5
        assert beh["archetype_primary"] == max(comp, key=comp.get)
        assert "mixed" not in comp
        behaviour = beh["behaviour"]
        assert behaviour["support_size_mean"] == pytest.approx(4.0, **FLOAT_TOL)
        assert behaviour["support_jaccard_lag1"] == pytest.approx(1.0, **FLOAT_TOL)
        assert "style_agreement_cosine" in behaviour
        assert "semantic_rotation_rate" in behaviour

    summary_path = tmp_path / "behavior_refresh_summary.json"
    assert summary_path.is_file()
    man = json.loads(summary_path.read_text(encoding="utf-8"))
    assert man["k_used"] == 5
    assert man["k_used_reason"] == "locked_five_named_archetypes"
    assert man["composition_stability"]["status"] in {"ok", "skipped"}


def test_build_policy_behavior_with_sleeve_produces_nonzero_tilts() -> None:
    """When sleeve_matrix is provided, tilt_trend/carry must be non-zero for non-EW."""
    rng = np.random.default_rng(42)
    K = 10
    weights = rng.dirichlet(np.ones(K) * 0.5, size=50)
    sleeve = np.zeros((K, 7))
    sleeve[:3, 0] = 1  # trend
    sleeve[3:6, 2] = 1  # carry
    sleeve[6:, 3] = 1  # defensive

    beh = build_policy_behavior(
        cell_id="test_sleeve",
        arm="eq",
        algo="ppo",
        weights=weights,
        sleeve_matrix=sleeve,
    )
    b = beh["behaviour"]
    assert b["tilt_trend"] != 0.0 or b["tilt_carry"] != 0.0 or b["tilt_defensive"] != 0.0


def test_behaviour_context_rehydrates_panel_and_sleeve(monkeypatch) -> None:
    """_behaviour_context_for_cell must rehydrate returns/sleeves when runner omits them."""
    from scripts.run_spectrum_campaign import _behaviour_context_for_cell

    T, K = 20, 6
    lake_rets = np.random.default_rng(0).normal(0, 0.01, size=(T, K))

    def _fake_lake(cfg, k=8):
        return list(range(T)), lake_rets, np.zeros((T, 4)), {"panel_source": "lake_sp500_sec"}

    monkeypatch.setattr(
        "src.eval.equity_substrate.load_lake_dyn_hrp_panel", _fake_lake
    )

    art = {
        "runner_artifact": {
            "paths": {"0": {"dates": list(range(T)), "weights": [[1.0 / K] * K] * T}},
        },
    }
    cfg = {
        "portfolio_arm": "eq",
        "train_world": "historical",
        "n_assets": K,
    }
    ctx = _behaviour_context_for_cell(art, cfg, art["runner_artifact"])
    assert ctx.get("asset_returns") is not None, "must rehydrate asset_returns from lake"
    assert np.asarray(ctx["asset_returns"]).shape[1] == K
    assert ctx.get("sleeve_matrix") is not None, "must supply spectrum sleeve_matrix fallback"
    assert np.asarray(ctx["sleeve_matrix"]).shape == (K, 7)
