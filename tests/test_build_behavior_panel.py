"""Tests for behaviour panel aggregator (A4)."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.build_behavior_panel import build_panel, discover_behavior_files


def test_discover_behavior_files_finds_policy_json(tmp_path: Path) -> None:
    src = tmp_path / "cells"
    src.mkdir()
    a = src / "cell_a_policy_behavior.json"
    a.write_text(json.dumps({"cell_id": "cell_a", "behaviour": {}}) + "\n")
    found = discover_behavior_files([src])
    assert len(found) == 1
    assert found[0].name == "cell_a_policy_behavior.json"


def test_build_panel_symlinks(tmp_path: Path) -> None:
    src = tmp_path / "src"
    out = tmp_path / "panel"
    src.mkdir()
    blob = src / "eq_policy_behavior.json"
    blob.write_text(json.dumps({"cell_id": "eq", "behaviour": {"turnover": 0.1}}) + "\n")
    manifest = build_panel(sources=[src], out_dir=out, link=True)
    assert manifest["n_cells"] == 1
    assert (out / "eq_policy_behavior.json").is_symlink()
    assert (out / "panel_manifest.json").is_file()
