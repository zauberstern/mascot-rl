"""Cell schema validation tests."""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.plumbing

from src.spectrum.cell_schema import validate_cell_cfg
from src.spectrum.yaml_loader import load_cell_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_all_cherrypick_yamls_pass_schema() -> None:
    base = ROOT / "config/spectrum/cherrypick"
    for path in list(base.glob("*.yaml")) + list((base / "narrative").glob("*.yaml")):
        validate_cell_cfg(load_cell_yaml(path), path=str(path))


def test_hybrid_and_featnet_yamls_pass_schema() -> None:
    for sub in (
        "cherrypick_hybrid",
        "cherrypick_featnet",
        "cherrypick_deskorg",
        "cherrypick_regime",
    ):
        base = ROOT / "config/spectrum" / sub
        if not base.is_dir():
            continue
        for path in base.glob("*.yaml"):
            validate_cell_cfg(load_cell_yaml(path), path=str(path))


def test_unknown_key_rejected() -> None:
    cfg = load_cell_yaml(
        ROOT / "config/spectrum/cherrypick/eq_K100_single_ppo_mlp_softmax_mean_std_cao.yaml"
    )
    cfg["bogus_key"] = 1
    with pytest.raises(ValueError, match="unknown keys"):
        validate_cell_cfg(cfg)


def test_alias_mismatch_rejected() -> None:
    cfg = load_cell_yaml(
        ROOT / "config/spectrum/cherrypick/eq_K100_single_ppo_mlp_softmax_mean_std_cao.yaml"
    )
    cfg["policy_algo"] = "not_ppo"
    with pytest.raises(ValueError, match="alias mismatch"):
        validate_cell_cfg(cfg)


def test_bootstrap_backend_and_fresh_quotes_optional() -> None:
    cfg = load_cell_yaml(
        ROOT / "config/spectrum/cherrypick/eq_K100_single_ppo_mlp_softmax_mean_std_cao.yaml"
    )
    cfg["bootstrap_backend"] = "arch"
    cfg["require_fresh_quotes"] = True
    validate_cell_cfg(cfg)


def test_bootstrap_backend_rejects_unknown() -> None:
    cfg = load_cell_yaml(
        ROOT / "config/spectrum/cherrypick/eq_K100_single_ppo_mlp_softmax_mean_std_cao.yaml"
    )
    cfg["bootstrap_backend"] = "bogus"
    with pytest.raises(ValueError):
        validate_cell_cfg(cfg)
