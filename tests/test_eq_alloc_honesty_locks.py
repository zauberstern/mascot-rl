"""Honesty locks: WFO is not CPCV; no capital-allocation claim fields."""
from __future__ import annotations

from pathlib import Path

import yaml

from src.reporting.capital_gates import assert_protocol_provenance


ROOT = Path(__file__).resolve().parents[1]
EQ_YAML = ROOT / "config" / "workflows" / "arm_equity.yaml"


def test_arm_equity_yaml_exists_and_is_eq():
    cfg = yaml.safe_load(EQ_YAML.read_text()) or {}
    assert cfg.get("arm", {}).get("id") == "eq"
    assert cfg.get("claim_label_stem") == "stk_ret"
    assert cfg.get("cost_in_decision") is True


def test_cpcv_artifact_spa_polarity_stamp():
    """Artifact-level lock: seed art must stamp policy_as_challenger."""
    art = {"spa_polarity": "policy_as_challenger", "spa_happo_as_claimant": True}
    assert art["spa_polarity"] == "policy_as_challenger"


def test_nested_wfo_alone_stays_non_capital_grade():
    """WFO is fine-tune, not CPCV. A report tagged only nested_wfo stays blocked."""
    report = {
        "eval_protocol": "nested_wfo",
        "alpha_found_historical": True,
        "nested_wfo": {"mode": "retrain_per_fold", "positive_fold_rate": 1.0},
    }
    out = assert_protocol_provenance(report)
    failures = (out.get("protocol_gate") or {}).get("gate_failures") or []
    assert any("protocol_not_capital_grade" in f for f in failures)
    assert out["protocol_gate"]["protocol_hygiene_ok"] is False
