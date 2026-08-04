"""PREREG §4 bakeoff decision rule must be evaluable from decision_fields."""
from __future__ import annotations

import pytest
from tests.conftest import FLOAT_TOL

import json
from pathlib import Path

from scripts.apply_prereg_bakeoff_decision import (
    NULL_REFERENCE_ARM,
    choose_winner,
    load_arm,
    normalize_decision_fields,
    qualifies,
)


def test_null_reference_default_is_dyn_hrp() -> None:
    assert NULL_REFERENCE_ARM == "dyn_hrp"


def test_qualifies_requires_collapse_std_and_binding() -> None:
    ok, reasons = qualifies(
        {
            "equal_weight_collapse_detected": False,
            "sharpe_std": 0.05,
            "turnover_cap_binding_fraction": 0.4,
        },
        projection_mode="hard",
    )
    assert ok is True
    assert reasons == []

    bad, reasons = qualifies(
        {
            "equal_weight_collapse_detected": True,
            "sharpe_std": 0.0001,
            "turnover_cap_binding_fraction": 0.0,
        },
        projection_mode="hard",
    )
    assert bad is False
    assert len(reasons) >= 2


def test_normalize_decision_fields_accepts_campaign_aliases() -> None:
    fields = normalize_decision_fields(
        {
            "equal_weight_collapse_detected_any": False,
            "sharpe_std_across_seeds": 0.001,
            "turnover_cap_binding_fraction_mean": 1.0,
            "l1_vs_ew_mean": 0.06,
        },
        path_summary={"sharpe_std": 0.002},
    )
    assert fields["equal_weight_collapse_detected"] is False
    assert fields["sharpe_std"] == pytest.approx(0.001, **FLOAT_TOL)
    assert fields["turnover_cap_binding_fraction"] == pytest.approx(1.0, **FLOAT_TOL)


def test_choose_winner_null_falls_back_to_dyn_hrp() -> None:
    decision = choose_winner(
        [
            {
                "arm": "dyn_liquidity",
                "qualifies": False,
                "sharpe_mean": 1.5,
                "decision_fields": {},
            }
        ]
    )
    assert decision["winner"] == "dyn_hrp"
    assert decision["status"] == "null_reference"
    assert decision.get("promotable") is False


def test_choose_winner_max_sharpe_among_eligible() -> None:
    decision = choose_winner(
        [
            {
                "arm": "dyn_liquidity",
                "qualifies": True,
                "sharpe_mean": 0.4,
                "decision_fields": {},
            },
            {
                "arm": "dyn_hrp",
                "qualifies": True,
                "sharpe_mean": 0.9,
                "decision_fields": {"equal_weight_collapse_detected": False},
            },
        ]
    )
    assert decision["winner"] == "dyn_hrp"
    assert decision["status"] == "promotable"


def test_load_arm_accepts_cfg_path_string(tmp_path: Path) -> None:
    summary = tmp_path / "cpcv_path_summary.json"
    summary.write_text(
        json.dumps(
            {
                "cfg": "config/workflows/arm_equity.yaml",
                "projection_mode": "hard",
                "confirmatory": {
                    "path_summary": {"sharpe_mean": 1.1, "sharpe_std": 0.001},
                    "decision_fields": {
                        "equal_weight_collapse_detected_any": False,
                        "sharpe_std_across_seeds": 0.001,
                        "turnover_cap_binding_fraction_mean": 1.0,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    row = load_arm(summary)
    assert row["arm"] == tmp_path.name
    assert row["qualifies"] is False
    assert row["projection_mode"] == "hard"
    assert "sharpe_std" in "".join(row["reject_reasons"])


def test_cli_writes_null_decision_with_hrp_reference(tmp_path: Path) -> None:
    from scripts.apply_prereg_bakeoff_decision import main

    root = tmp_path / "bakeoff"
    for arm, sharpe, std in (
        ("dyn_hrp", 1.1, 0.001),
        ("dyn_liquidity", 0.8, 0.001),
        ("dyn_crucible", 0.6, 0.0003),
    ):
        arm_dir = root / arm
        arm_dir.mkdir(parents=True)
        (arm_dir / "cpcv_path_summary.json").write_text(
            json.dumps(
                {
                    "cfg": "config/workflows/arm_equity.yaml",
                    "projection_mode": "hard",
                    "confirmatory": {
                        "path_summary": {"sharpe_mean": sharpe, "sharpe_std": std},
                        "decision_fields": {
                            "equal_weight_collapse_detected_any": arm == "dyn_crucible",
                            "sharpe_std_across_seeds": std,
                            "turnover_cap_binding_fraction_mean": 1.0,
                        },
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
    out = tmp_path / "decision.json"
    ledger = tmp_path / "ledger.json"
    yaml_path = tmp_path / "spine.yaml"
    yaml_path.write_text("universe_arm: dyn_liquidity\nprojection_mode: hard\n")
    code = main(
        [
            "--bakeoff-root",
            str(root),
            "--arms",
            "dyn_hrp,dyn_liquidity,dyn_crucible",
            "--out",
            str(out),
            "--ledger",
            str(ledger),
            "--promote-yaml",
            "--yaml",
            str(yaml_path),
        ]
    )
    assert code == 0
    blob = json.loads(out.read_text(encoding="utf-8"))
    assert blob["decision"]["status"] == "null_reference"
    assert blob["decision"]["winner"] == "dyn_hrp"
    assert "dyn_crucible" in blob["decision"].get("deferred_arms", ["dyn_crucible"])
    assert "universe_arm: dyn_hrp" in yaml_path.read_text()
