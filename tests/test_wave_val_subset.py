"""VAL wave subset coverage and wiring tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.plumbing

from src.aws_burst.profiles import PRODUCTION_WAVES
from src.aws_burst.waves import WAVES, discover_wave_cells
from src.eval.universe_fingerprint import EQ_BURST_WAVES
from src.spectrum.yaml_loader import load_cell_yaml

ROOT = Path(__file__).resolve().parents[1]
VAL_DIR = ROOT / "config" / "spectrum" / "cherrypick_val"
MANIFEST = VAL_DIR / "manifest.json"
MAMBA_K100 = (
    ROOT
    / "config"
    / "spectrum"
    / "cherrypick"
    / "_dropped_mamba"
    / "eq_K100_single_ppo_mamba_softmax_mean_std_cao.yaml"
)


def test_val_wave_registered() -> None:
    assert "VAL" in WAVES
    assert "VAL" in PRODUCTION_WAVES
    assert "VAL" in EQ_BURST_WAVES


def test_val_cell_count_and_manifest_sync() -> None:
    cells = discover_wave_cells(ROOT, "VAL")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    stems = {Path(c).stem for c in cells}
    assert len(cells) == int(manifest["n_cells"]) == 90
    assert stems == set(manifest["cells"])


def test_val_yaml_load_valid() -> None:
    for path in sorted(VAL_DIR.glob("*.yaml")):
        cfg = load_cell_yaml(path)
        assert cfg.get("portfolio_arm") == "eq"
        assert int(cfg.get("n_assets") or 0) in {10, 25, 100, 200}


def test_val_axis_coverage() -> None:
    stems = {Path(c).stem for c in discover_wave_cells(ROOT, "VAL")}
    algos = {"ppo", "cppo", "sac", "td3", "ddpg", "dqn", "mcpg", "rrl", "happo"}
    found_algos = {a for a in algos if any(f"_{a}_" in s or f"multi_{a}_" in s for s in stems)}
    assert found_algos == algos
    for body in ("gru", "lstm", "transformer", "mamba"):
        assert any(f"_{body}_" in s for s in stems), body
    for head in ("tanh_l1", "dirichlet_tilt", "dirichlet_mean"):
        assert any(head.replace("_", "_") in s for s in stems), head
    assert any("rb-sb3" in s for s in stems)
    assert any("hybrid_pretrain_finetune" in s for s in stems)
    assert any("_K10_" in s for s in stems)
    assert any("_K25_" in s for s in stems)
    assert any("_K200_" in s for s in stems)


def test_mamba_probes_only_change_k() -> None:
    k100 = MAMBA_K100.read_text(encoding="utf-8")
    for k in (10, 25):
        stem = f"eq_K{k}_single_ppo_mamba_softmax_mean_std_cao"
        probe = (VAL_DIR / f"{stem}.yaml").read_text(encoding="utf-8")
        assert f"n_assets: {k}" in probe
        assert f"spectrum_cell_id: {stem}" in probe
        # Same algo/body/head/objective as K100 source aside from K.
        assert "architecture: mamba" in probe
        assert "weight_head: softmax" in probe
        assert "n_minibatches: 32" in (VAL_DIR / "eq_K10_single_ppo_mamba_softmax_mean_std_cao.yaml").read_text()
        assert "n_minibatches: 64" in (VAL_DIR / "eq_K25_single_ppo_mamba_softmax_mean_std_cao.yaml").read_text()
        assert "use_equity_feature_cube: true" in (VAL_DIR / "eq_K10_single_ppo_mamba_softmax_mean_std_cao.yaml").read_text()
        assert "requires_himem: true" in (VAL_DIR / "eq_K10_single_ppo_mamba_softmax_mean_std_cao.yaml").read_text()


def test_s3_watch_uses_all_burst_profiles() -> None:
    from scripts import aws_burst_s3_watch as watch_mod
    from src.aws_burst.profiles import BURST_PROFILES

    assert len(watch_mod.PROFILE_ACCOUNT_IDS) == len(BURST_PROFILES) == 4
    assert "volsurf-burst-4" in watch_mod.PROFILE_ACCOUNT_IDS
