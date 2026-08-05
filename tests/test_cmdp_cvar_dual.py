"""CMDP CVaR cost signal feeds per-step costs."""
from __future__ import annotations

import pytest
from tests.conftest import FLOAT_TOL
import torch

from mascotrl.policy.cmdp_config import build_step_costs, resolve_cmdp_cfg


def test_resolve_cmdp_cfg_from_yaml_block():
    cfg = {"cmdp": {"enabled": True, "cost_signal": "cvar", "target": 0.1, "alpha": 0.9}}
    out = resolve_cmdp_cfg(cfg)
    assert out["cmdp_enabled"] is True
    assert out["cmdp_cost_signal"] == "cvar"
    assert out["cmdp_limit_d"] == pytest.approx(0.1, **FLOAT_TOL)


def test_cvar_cost_is_downside_loss():
    rewards = torch.tensor([0.1, -0.2, 0.05])
    costs = build_step_costs(rewards, signal="cvar")
    assert costs[0].item() == pytest.approx(0.0, **FLOAT_TOL)
    assert costs[1].item() == pytest.approx(0.2)
