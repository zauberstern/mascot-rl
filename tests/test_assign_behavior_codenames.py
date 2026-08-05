"""D-1: assign_behavior_codenames fail-closed (--expect-stems, imputation cap)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "assign_behavior_codenames.py"
sys.path.insert(0, str(ROOT))

from scripts.assign_behavior_codenames import cluster_behaviours, main  # noqa: E402
from mascotrl.reporting.behavior_metrics import BEHAVIOUR_MEASURE_IDS  # noqa: E402


def _write_behavior(path: Path, measures: dict[str, float] | None = None) -> None:
    beh = {"behaviour": measures or {"hhi_mean": 0.2, "turnover_mean": 0.1}}
    path.write_text(json.dumps(beh), encoding="utf-8")


def test_main_returns_nonzero_when_cluster_not_ok(tmp_path: Path) -> None:
    out = tmp_path / "codenames.json"
    code = main(
        [
            "--dir",
            str(tmp_path / "empty"),
            "--out",
            str(out),
        ]
    )
    assert code != 0


def test_expect_stems_refuses_missing_policy_behavior(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cells": ["cell_a", "cell_b"],
                "dropped_cells": [],
            }
        ),
        encoding="utf-8",
    )
    beh_dir = tmp_path / "beh"
    beh_dir.mkdir()
    _write_behavior(beh_dir / "cell_a_policy_behavior.json")

    out = tmp_path / "out.json"
    code = main(
        [
            "--dir",
            str(beh_dir),
            "--out",
            str(out),
            "--expect-stems",
            str(manifest),
        ]
    )
    assert code != 0


def test_imputation_fraction_over_half_fails_closed(tmp_path: Path) -> None:
    beh_dir = tmp_path / "beh"
    beh_dir.mkdir()
    # Empty behaviour imputes every measure -> 100% imputed.
    (beh_dir / "sparse_cell_policy_behavior.json").write_text(
        json.dumps({"behaviour": {}}),
        encoding="utf-8",
    )
    paths = sorted(beh_dir.glob("*_policy_behavior.json"))
    payload = cluster_behaviours(paths, k=1)
    assert payload.get("ok") is False
    assert "imputation" in str(payload.get("reason") or "").lower()
    meta = payload.get("meta") or {}
    imputation = meta.get("imputation") or {}
    assert imputation.get("max_fraction", 0) > 0.5
    cells = payload.get("cells") or []
    if cells:
        assert cells[0].get("imputation_fraction", 0) > 0.5


def test_vector_imputation_meta_counts_missing_keys(tmp_path: Path) -> None:
    beh_dir = tmp_path / "beh"
    beh_dir.mkdir()
    # Keep imputation below the 50% fail-closed threshold (23 measures).
    n_present = len(BEHAVIOUR_MEASURE_IDS) - 10
    measures = {BEHAVIOUR_MEASURE_IDS[i]: 0.1 for i in range(n_present)}
    _write_behavior(beh_dir / "partial_policy_behavior.json", measures)
    paths = sorted(beh_dir.glob("*_policy_behavior.json"))
    payload = cluster_behaviours(paths, k=1)
    assert payload.get("ok") is True
    cells = payload.get("cells") or []
    expected_frac = 1.0 - (n_present / len(BEHAVIOUR_MEASURE_IDS))
    assert abs(cells[0]["imputation_fraction"] - expected_frac) < 1e-9


def test_cli_expect_stems_success(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cells": ["ok_cell"]}), encoding="utf-8")
    beh_dir = tmp_path / "beh"
    beh_dir.mkdir()
    measures = {mid: 0.05 for mid in BEHAVIOUR_MEASURE_IDS}
    _write_behavior(beh_dir / "ok_cell_policy_behavior.json", measures)
    out = tmp_path / "codenames.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dir",
            str(beh_dir),
            "--out",
            str(out),
            "--expect-stems",
            str(manifest),
            "--k",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload.get("ok") is True
