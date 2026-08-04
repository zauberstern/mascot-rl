"""Friction honesty locks: XOR OM-touch, cost_multiplier, no soft-fee collapse."""
from __future__ import annotations

import numpy as np
import pytest
from tests.conftest import FLOAT_TOL
import torch

from src.arms import ArmSpec
from src.eval.friction import FrictionSpec, apply_costs


def test_apply_costs_never_nan_valid_inputs():
    w = torch.tensor([[0.4, -0.3, 0.2]])
    w_prev = torch.zeros(1, 3)
    ret = torch.tensor([0.01, -0.02, 0.005])
    out = apply_costs(w, w_prev, ret, equity_bps=5.0)
    assert np.isfinite(out.gross)
    assert np.isfinite(out.net)
    assert np.isfinite(out.equity_spread)


def test_equity_and_option_spreads_separate_no_double_count():
    arm = ArmSpec(id="mix", option_slots=1, equity_slots=1, delta_mode="joint")
    w = torch.tensor([[0.5, 0.5]])
    w_prev = torch.zeros(1, 2)
    ret = torch.tensor([0.01, 0.01])
    hs = torch.tensor([[0.1, 99.0]])
    cb = torch.tensor([[50.0, 1.0]])
    out = apply_costs(
        w,
        w_prev,
        ret,
        arm=arm,
        half_spread=hs,
        capital_base=cb,
        om_touch_enabled=True,
        equity_bps=10.0,
        hedge_leg_bps=0.0,
    )
    assert out.option_spread > 0.0
    assert out.equity_spread > 0.0
    total_drag = out.gross - out.net
    borrow = float(getattr(out, "borrow", 0.0) or 0.0)
    assert total_drag == pytest.approx(
        out.option_spread + out.equity_spread + out.hedge_leg + out.funding + borrow,
        abs=1e-9,
    )


def test_om_touch_xor_stylized_execution():
    arm = ArmSpec(id="opt", option_slots=2, equity_slots=0, delta_mode="off")
    w = torch.tensor([[0.6, 0.4]])
    w_prev = torch.zeros(1, 2)
    ret = torch.tensor([0.0, 0.0])
    hs = torch.tensor([[0.05, 0.05]])
    cb = torch.tensor([[100.0, 100.0]])
    with_touch = apply_costs(
        w,
        w_prev,
        ret,
        arm=arm,
        half_spread=hs,
        capital_base=cb,
        om_touch_enabled=True,
        execution_spread_bps=50.0,
        execution_impact_coef=1.0,
    )
    stylized = apply_costs(
        w,
        w_prev,
        ret,
        arm=arm,
        half_spread=hs,
        capital_base=cb,
        om_touch_enabled=False,
        execution_spread_bps=50.0,
        execution_impact_coef=1.0,
    )
    assert with_touch.option_spread > 0.0
    assert stylized.option_spread > 0.0
    assert with_touch.option_spread != pytest.approx(stylized.option_spread)


def test_execution_impact_coef_option_only_not_equity():
    arm_eq = ArmSpec(id="eq", option_slots=0, equity_slots=2, delta_mode="off")
    w = torch.tensor([[0.5, -0.5]])
    w_prev = torch.zeros(1, 2)
    ret = torch.tensor([0.01, 0.01])
    out_eq = apply_costs(
        w,
        w_prev,
        ret,
        arm=arm_eq,
        equity_bps=0.0,
        impact_c_eq=0.0,
        execution_impact_coef=10.0,
        om_touch_enabled=False,
    )
    assert out_eq.option_spread == pytest.approx(0.0, **FLOAT_TOL)
    assert out_eq.equity_spread == pytest.approx(0.0, **FLOAT_TOL)


def test_cost_multiplier_scales_costs_not_gross():
    arm = ArmSpec(id="eq", option_slots=0, equity_slots=2, delta_mode="off")
    w = torch.tensor([[0.7, 0.3]])
    w_prev = torch.zeros(1, 2)
    ret = torch.tensor([0.02, -0.01])
    fr1 = FrictionSpec(equity_bps=10.0, cost_multiplier=1.0, om_touch_enabled=False)
    fr2 = FrictionSpec(equity_bps=10.0, cost_multiplier=3.0, om_touch_enabled=False)
    a = apply_costs(w, w_prev, ret, arm=arm, friction=fr1)
    b = apply_costs(w, w_prev, ret, arm=arm, friction=fr2)
    assert a.gross == pytest.approx(b.gross)
    assert b.equity_spread == pytest.approx(3.0 * a.equity_spread)


def test_no_soft_fee_collapse_costs_deducted_from_pnl():
    """Honesty lock: net = gross - costs; costs are not a soft reward penalty."""
    w = torch.tensor([[1.0]])
    w_prev = torch.zeros(1, 1)
    ret = torch.tensor([0.05])
    out = apply_costs(
        w,
        w_prev,
        ret,
        arm=ArmSpec(id="eq", option_slots=0, equity_slots=1, delta_mode="off"),
        equity_bps=100.0,
        om_touch_enabled=False,
    )
    assert out.net < out.gross
    assert out.net != out.gross
