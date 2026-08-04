"""Regression tests for scripts/validate_h0_summary.py (roadmap W2 checklist)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate_h0_summary import validate_summary


def _minimal_summary(*, k: int = 100, n_seeds: int = 10) -> dict:
    per_seed = {str(i): {"sharpe": 0.1} for i in range(n_seeds)}
    return {
        "k": k,
        "universe_arm": "dyn_hrp",
        "confirmatory": {
            "path_summary": {
                "n_seeds": n_seeds,
                "sharpe_mean": 0.12,
                "sharpe_std": 0.05,
                "per_seed": per_seed,
            },
            "gates": {
                "gate1": {"pass": True},
                "gate2": {"pass": True},
                "gate3": {"pass": False},
            },
            "benchmark_sharpes": {"equal_weight": 0.08},
            "stats_table": {
                "deflated_sharpe": 0.95,
                "hansen_spa_vs_ew": 0.04,
            },
            "negative_controls": {
                "pipeline_broken": False,
                "shuffled": {"sharpe": -0.02},
                "permuted": {"sharpe": 0.0},
                "date_shifted": {"sharpe": 0.01},
            },
        },
        "policy_behavior": {"feeds_capital_gates": False},
        "book": {"out_dir": "/tmp/book"},
    }


def test_validate_summary_ok(tmp_path: Path) -> None:
    p = tmp_path / "cpcv_path_summary.json"
    p.write_text(json.dumps(_minimal_summary()), encoding="utf-8")
    out = validate_summary(p, label="HEAD-EQ")
    assert out["ok"] is True
    assert out["errors"] == []


def test_validate_summary_ok_per_seed_list(tmp_path: Path) -> None:
    body = _minimal_summary()
    body["confirmatory"]["path_summary"]["per_seed"] = [0.1] * 10
    p = tmp_path / "cpcv_path_summary.json"
    p.write_text(json.dumps(body), encoding="utf-8")
    out = validate_summary(p, label="HEAD-EQ")
    assert out["ok"] is True


def test_validate_summary_k_mismatch(tmp_path: Path) -> None:
    p = tmp_path / "cpcv_path_summary.json"
    p.write_text(json.dumps(_minimal_summary(k=432)), encoding="utf-8")
    out = validate_summary(p, label="HEAD-EQ", require_k=100)
    assert out["ok"] is False
    assert any("k_mismatch" in e for e in out["errors"])


def test_validate_summary_refuses_legacy_capital_claim_true(tmp_path: Path) -> None:
    body = _minimal_summary()
    body["capital_claim_allowed"] = True
    p = tmp_path / "cpcv_path_summary.json"
    p.write_text(json.dumps(body), encoding="utf-8")
    out = validate_summary(p, label="HEAD-EQ")
    assert out["ok"] is False
    assert any("legacy_capital_claim" in e for e in out["errors"])



def test_validate_summary_missing_file(tmp_path: Path) -> None:
    out = validate_summary(tmp_path / "missing.json", label="HEAD-EQ")
    assert out["ok"] is False
    assert out["errors"] == ["missing_file"]
