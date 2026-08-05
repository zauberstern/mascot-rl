"""Tests for spectrum ArmSpec."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from mascotrl.arms import ArmSpec, arm_spec_from_cfg, default_arm_spec


def test_default_arm_matches_status_quo_options_only():
    arm = default_arm_spec(50)
    assert arm.id == "opt"
    assert arm.option_slots == 50
    assert arm.equity_slots == 0
    assert arm.n_slots == 50
    assert arm.delta_mode == "soft"
    assert arm.option_index().tolist() == list(range(50))
    assert arm.equity_index().size == 0


def test_absent_arm_block_resolves_to_opt():
    arm = arm_spec_from_cfg({"n_assets": 50})
    assert arm == default_arm_spec(50)


def test_arm_eq_and_mix_layouts():
    eq = ArmSpec(id="eq", option_slots=0, equity_slots=50, delta_mode="off")
    assert eq.n_slots == 50
    assert eq.option_index().size == 0
    assert eq.equity_index().tolist() == list(range(50))
    assert eq.delta_vector().tolist() == [1.0] * 50

    mix = ArmSpec(id="mix", option_slots=50, equity_slots=50, delta_mode="joint")
    assert mix.n_slots == 100
    d = mix.delta_vector(option_deltas=np.linspace(0.3, 0.7, 50))
    assert d.shape == (100,)
    assert np.allclose(d[50:], 1.0)
    assert np.allclose(d[:50], np.linspace(0.3, 0.7, 50))


def test_arm_yaml_configs_parse():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config" / "workflows" / "arm_options.yaml").read_text())
    arm = arm_spec_from_cfg(cfg)
    assert arm.id == "opt"
    assert arm.n_slots == 50
    assert arm.delta_mode == "soft"


def test_invalid_arm_rejected():
    with pytest.raises(ValueError):
        ArmSpec(id="opt", option_slots=50, equity_slots=1, delta_mode="soft")
    with pytest.raises(ValueError):
        ArmSpec(id="eq", option_slots=1, equity_slots=50, delta_mode="off")
    with pytest.raises(ValueError):
        ArmSpec(id="mix", option_slots=0, equity_slots=50, delta_mode="joint")
