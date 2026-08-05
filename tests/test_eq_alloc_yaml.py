"""arm_* workflow YAML locks (equity / options / mix spectrum spines)."""
from __future__ import annotations

from pathlib import Path

import pytest
from tests.conftest import FLOAT_TOL
import yaml

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "config" / "workflows"


def _load(name: str) -> dict:
    p = WF / name
    assert p.is_file(), p
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def test_arm_equity_locks():
    cfg = _load("arm_equity.yaml")
    assert cfg["arm"]["id"] == "eq"
    assert int(cfg["arm"]["equity_slots"]) == 50
    assert cfg["primary_train"] == "historical_arm_env"
    assert cfg["claim_label_stem"] == "stk_ret"
    assert float(cfg["equity_bps"]) == pytest.approx(5.0, **FLOAT_TOL)
    assert int(cfg["cpcv_n_splits"]) == 8
    assert int(cfg["cpcv_n_test_groups"]) == 3
    assert int(cfg["cpcv_purge_days"]) == 21
    assert int(cfg["cpcv_embargo_days"]) == 21
    assert cfg["cost_in_decision"] is True
    assert cfg["portfolio_arm"] == "eq"
    assert cfg["train_world"] == "historical"


def test_arm_options_and_mix_exist():
    opt = _load("arm_options.yaml")
    mix = _load("arm_mix.yaml")
    assert opt["arm"]["id"] == "opt" or opt.get("portfolio_arm") == "opt"
    assert mix["arm"]["id"] == "mix" or mix.get("portfolio_arm") == "mix"


def test_arm_equity_passes_turnover_honesty() -> None:
    from mascotrl.eval.yaml_honesty import assert_turnover_cap_honesty

    cfg = _load("arm_equity.yaml")
    out = assert_turnover_cap_honesty(cfg)
    assert out["projection_mode"] == "hard"
    assert out["turnover_cap_enforced"] is True


def test_eq_alloc_refuses_decorative_turnover_limit_under_soft():
    """turnover_limit may only be advertised when projection_mode is hard."""
    from mascotrl.eval.yaml_honesty import assert_turnover_cap_honesty

    assert_turnover_cap_honesty({"projection_mode": "hard", "turnover_limit": 0.15})
    with pytest.raises(AssertionError, match="turnover_limit"):
        assert_turnover_cap_honesty({"projection_mode": "soft", "turnover_limit": 0.15})
    assert_turnover_cap_honesty({"projection_mode": "soft"})
