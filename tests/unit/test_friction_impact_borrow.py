"""Phase G: impact_c_eq sqrt participation + borrow_bps_annual on shorts."""
from __future__ import annotations

import pytest
from tests.conftest import FLOAT_TOL
import torch

from src.arms import ArmSpec
from src.eval.friction import FrictionSpec, apply_costs


def test_impact_c_eq_increases_equity_costs():
    arm = ArmSpec(id="eq", option_slots=0, equity_slots=2, delta_mode="off")
    w = torch.tensor([[0.5, -0.5]])
    w_prev = torch.zeros(1, 2)
    ret = torch.tensor([0.01, 0.02])
    base = apply_costs(
        w, w_prev, ret, arm=arm, equity_bps=5.0, impact_c_eq=0.0, hedge_leg_bps=0.0
    )
    with_impact = apply_costs(
        w, w_prev, ret, arm=arm, equity_bps=5.0, impact_c_eq=0.5, hedge_leg_bps=0.0
    )
    assert with_impact.equity_spread > base.equity_spread
    # ADV missing → unit-notional proxy; impact = (c * sum sqrt(|dw|)) / 1e4.
    dw = (w - w_prev).abs().reshape(-1)
    expected_extra = (0.5 * float(torch.sqrt(dw).sum().item())) / 1e4
    assert with_impact.equity_spread == pytest.approx(base.equity_spread + expected_extra)


def test_borrow_bps_annual_charges_short_weights():
    arm = ArmSpec(id="eq", option_slots=0, equity_slots=2, delta_mode="off")
    w = torch.tensor([[0.5, -0.5]])
    w_prev = torch.zeros(1, 2)
    ret = torch.tensor([0.0, 0.0])
    no_borrow = apply_costs(
        w, w_prev, ret, arm=arm, equity_bps=0.0, impact_c_eq=0.0, hedge_leg_bps=0.0
    )
    with_borrow = apply_costs(
        w,
        w_prev,
        ret,
        arm=arm,
        equity_bps=0.0,
        impact_c_eq=0.0,
        hedge_leg_bps=0.0,
        borrow_bps_annual=25.0,
        borrow_dt_years=1.0 / 252.0,
    )
    expected = (25.0 / 1e4) * (1.0 / 252.0) * 0.5
    assert with_borrow.funding == pytest.approx(expected)
    assert with_borrow.funding > no_borrow.funding
    assert with_borrow.net < no_borrow.net


def test_borrow_bps_alias_and_long_only_zero():
    arm = ArmSpec(id="eq", option_slots=0, equity_slots=2, delta_mode="off")
    w = torch.tensor([[0.4, 0.6]])
    w_prev = torch.zeros(1, 2)
    ret = torch.tensor([0.0, 0.0])
    out = apply_costs(
        w, w_prev, ret, arm=arm, equity_bps=0.0, borrow_bps=100.0, impact_c_eq=0.0
    )
    assert out.funding == pytest.approx(0.0, **FLOAT_TOL)


def test_friction_spec_impact_overrides_kwarg():
    arm = ArmSpec(id="eq", option_slots=0, equity_slots=2, delta_mode="off")
    w = torch.tensor([[1.0, 0.0]])
    w_prev = torch.zeros(1, 2)
    ret = torch.tensor([0.0, 0.0])
    fr = FrictionSpec(equity_bps=0.0, impact_c_eq=1.0, om_touch_enabled=False)
    out = apply_costs(w, w_prev, ret, arm=arm, friction=fr, impact_c_eq=0.0)
    # c * sqrt(|dw|) / 1e4 = 1 * 1 / 1e4
    assert out.equity_spread == pytest.approx(1.0 / 1e4)


def test_impact_c_eq_full_rebalance_is_not_nav_destroying():
    """Guardrail: impact_c_eq=0.5 must not charge ~50% NAV without ADV."""
    arm = ArmSpec(id="eq", option_slots=0, equity_slots=2, delta_mode="off")
    w = torch.tensor([[0.5, 0.5]])
    w_prev = torch.zeros(1, 2)
    ret = torch.tensor([0.0, 0.0])
    out = apply_costs(
        w, w_prev, ret, arm=arm, equity_bps=5.0, impact_c_eq=0.5, hedge_leg_bps=0.0
    )
    turn = float((w - w_prev).abs().sum().item())
    impact = (0.5 * float(torch.sqrt((w - w_prev).abs().reshape(-1)).sum().item())) / 1e4
    expected = (5.0 / 1e4) * turn + impact
    assert out.equity_spread == pytest.approx(expected)
    assert out.equity_spread < 0.01
