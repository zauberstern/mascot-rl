"""Tests for scripts/validate_val_subset.py."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_val_subset import (
    HYBRID_STEM,
    validate_hybrid_error,
    validate_val_artifact,
    validate_val_dir,
)

ROOT = Path(__file__).resolve().parents[1]


def _minimal_art(**overrides) -> dict:
    base = {
        "compute_host": "remote",
        "instance_type": "m7i-flex.large",
        "container_digest": "sha256:abc",
        "requirements_lock_sha256": "deadbeef",
        "universe_fingerprint": "fp1",
        "universe_fingerprint_kind": "panel_bundle_sha256",
        "panel_source": "lake_sp500_sec",
        "weight_head": "softmax",
        "algo": "ppo",
        "k": 100,
        "n_assets": 100,
        "cpcv": {"n_splits": 6, "n_test_groups": 2, "purge_days": 21, "embargo_days": 21},
        "final_weights": [[0.5, 0.5]],
        "metrics": {"sharpe": 0.1},
    }
    base.update(overrides)
    return base


def test_validate_val_artifact_green() -> None:
    manifest = json.loads(
        (ROOT / "config/spectrum/cherrypick_val/manifest.json").read_text(encoding="utf-8")
    )
    stem = "eq_K100_single_ppo_mlp_softmax_mean_std_cao"
    res = validate_val_artifact(stem, _minimal_art(), manifest=manifest)
    assert res["ok"] is True


def test_validate_val_artifact_rejects_legacy_capital_claim_true() -> None:
    manifest = json.loads(
        (ROOT / "config/spectrum/cherrypick_val/manifest.json").read_text(encoding="utf-8")
    )
    stem = "eq_K100_single_ppo_mlp_softmax_mean_std_cao"
    res = validate_val_artifact(
        stem, _minimal_art(capital_claim_allowed=True), manifest=manifest
    )
    assert res["ok"] is False




def test_mamba_probe_k_from_stem() -> None:
    manifest = json.loads(
        (ROOT / "config/spectrum/cherrypick_val/manifest.json").read_text(encoding="utf-8")
    )
    stem = "eq_K10_single_ppo_mamba_softmax_mean_std_cao"
    art = _minimal_art()
    art.pop("k", None)
    art.pop("n_assets", None)
    res = validate_val_artifact(stem, art, manifest=manifest)
    assert res["ok"] is True, res.get("errors")


def test_hybrid_error_acceptance() -> None:
    err = {
        "n_attempts": 1,
        "reason": "fail-closed non-finite policy_obs across CPCV folds",
    }
    res = validate_hybrid_error(err)
    assert res["ok"] is True


def test_validate_val_dir_missing_reports_anomaly(tmp_path: Path) -> None:
    manifest = json.loads(
        (ROOT / "config/spectrum/cherrypick_val/manifest.json").read_text(encoding="utf-8")
    )
    # Write one passing cell only
    stem = manifest["cells"][0]
    (tmp_path / f"{stem}.json").write_text(
        json.dumps(_minimal_art(spectrum_cell_id=stem)), encoding="utf-8"
    )
    res = validate_val_dir(tmp_path, root=ROOT)
    assert res["ok"] is False
    assert res["n_anomalies"] >= 1
