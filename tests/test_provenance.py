"""Run provenance manifest and report schema validation (W8)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mascotrl.reporting.provenance import (
    REQUIRED_CLAIM_FIELDS,
    REQUIRED_REPORT_FIELDS,
    RunManifest,
    config_hash,
    validate_report_schema,
)


def test_config_hash_is_stable_and_order_independent():
    a = {"seed": 42, "n_assets": 50, "lr": 3e-4}
    b = {"lr": 3e-4, "n_assets": 50, "seed": 42}
    assert config_hash(a) == config_hash(b)


def test_config_hash_changes_with_content():
    assert config_hash({"seed": 1}) != config_hash({"seed": 2})


def test_manifest_captures_seeds_and_protocol_switches():
    cfg = {
        "seed": 42,
        "eval_seeds": "0,1,2",
        "nested_wfo_retrain": True,
        "train_distribution": "optionmetrics_only",
        "capital_gates_require_factor_alpha": True,
        "min_break_even_spread_multiplier": 0.25,
        "hedge_frequency": "daily",
    }
    m = RunManifest(run_label="test-run", cfg=cfg).build()
    assert m["seeds"]["seed"] == 42
    assert m["seeds"]["eval_seeds"] == "0,1,2"
    assert m["protocol_switches"]["train_distribution"] == "optionmetrics_only"
    assert m["protocol_switches"]["hedge_frequency"] == "daily"
    assert m["config_sha256"] == config_hash(cfg)


def test_manifest_records_environment_and_code_state():
    m = RunManifest(run_label="r", cfg={}).build()
    assert "python" in m["environment"]
    assert "numpy" in m["environment"]["packages"]
    assert "git_dirty" in m["code"]
    assert isinstance(m["code"]["git_dirty"], bool)


def test_manifest_writes_json_artifact(tmp_path: Path):
    path = RunManifest(run_label="r1", cfg={"seed": 7}).write(tmp_path)
    assert path.name == "RUN_MANIFEST.json"
    payload = json.loads(path.read_text())
    assert payload["run_label"] == "r1"
    assert payload["seeds"]["seed"] == 7


def test_manifest_warns_that_dirty_tree_is_not_reproducible():
    m = RunManifest(run_label="r", cfg={}).build()
    assert "not reproducible" in m["reproduction"]["note"]


# --------------------------------------------------------------- schema checks

def _complete_report() -> dict:
    rep = {f: "x" for f in REQUIRED_REPORT_FIELDS}
    rep["n_assets"] = 50
    for f in REQUIRED_CLAIM_FIELDS:
        rep[f] = {"present": True}
    return rep


def test_schema_passes_on_complete_report():
    out = validate_report_schema(_complete_report(), require_claim_fields=True)
    assert out["ok"] is True
    assert out["missing_required"] == []
    assert out["missing_claim_evidence"] == []


def test_schema_flags_missing_required_field():
    rep = _complete_report()
    del rep["eval_protocol"]
    out = validate_report_schema(rep)
    assert out["ok"] is False
    assert "eval_protocol" in out["missing_required"]


def test_schema_flags_missing_claim_evidence():
    """A capital claim without factor alpha or a cost ladder is incomplete."""
    rep = _complete_report()
    del rep["factor_alpha"]
    del rep["cost_ladder"]
    out = validate_report_schema(rep, require_claim_fields=True)
    assert out["ok"] is False
    assert "factor_alpha" in out["missing_claim_evidence"]
    assert "cost_ladder" in out["missing_claim_evidence"]


def test_schema_claim_fields_not_required_when_no_claim():
    rep = {f: "x" for f in REQUIRED_REPORT_FIELDS}
    rep["n_assets"] = 50
    out = validate_report_schema(rep, require_claim_fields=False)
    assert out["ok"] is True


def test_schema_strict_mode_raises():
    with pytest.raises(RuntimeError, match="schema incomplete"):
        validate_report_schema({}, strict=True)


def test_claim_fields_cover_the_new_evidence_requirements():
    """The paper's claim depends on these; locking the list prevents silent drops."""
    assert "deflated_sharpe_oos" in REQUIRED_CLAIM_FIELDS
    assert "hac_inference_oos" in REQUIRED_CLAIM_FIELDS
    assert "factor_alpha" in REQUIRED_CLAIM_FIELDS
    assert "cost_ladder" in REQUIRED_CLAIM_FIELDS
    assert "n_trials_breakdown" in REQUIRED_CLAIM_FIELDS
