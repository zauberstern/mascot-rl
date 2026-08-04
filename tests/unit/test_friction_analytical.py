"""Analytical boundary tests for apply_costs (confirmatory-critical friction math)."""
from __future__ import annotations

import pytest
import torch

from src.arms import ArmSpec
from src.eval.friction import apply_costs
from tests.conftest import FLOAT_TOL


def _eq_arm(k: int = 4) -> ArmSpec:
    return ArmSpec(id="eq", option_slots=0, equity_slots=k, delta_mode="off")


@pytest.mark.unit
def test_zero_turnover_zero_cost():
    """Holding weights fixed must charge zero equity spread."""
    w = torch.tensor([0.25, 0.25, 0.25, 0.25])
    ret = torch.tensor([0.01, -0.01, 0.0, 0.005])
    bd = apply_costs(
        w, w, ret, arm=_eq_arm(), equity_bps=5.0, om_touch_enabled=False
    )
    assert bd.equity_spread == pytest.approx(0.0, **FLOAT_TOL)
    assert bd.option_spread == pytest.approx(0.0, **FLOAT_TOL)
    assert bd.hedge_leg == pytest.approx(0.0, **FLOAT_TOL)


@pytest.mark.unit
def test_known_spread_on_hand_computed_transition():
    """Hand-computed |Δw| * equity_bps / 1e4 matches apply_costs equity_spread."""
    w_prev = torch.tensor([0.5, 0.5, 0.0, 0.0])
    w = torch.tensor([0.0, 0.0, 0.5, 0.5])
    ret = torch.zeros(4)
    bps = 5.0
    bd = apply_costs(
        w, w_prev, ret, arm=_eq_arm(), equity_bps=bps, om_touch_enabled=False
    )
    # L1 turnover = |0.5|+|0.5|+|0.5|+|0.5| = 2.0; cost = 2.0 * 5/1e4
    expected = 2.0 * (bps / 1e4)
    assert bd.equity_spread == pytest.approx(expected, **FLOAT_TOL)


@pytest.mark.unit
def test_borrow_cost_on_short_exposure():
    """Annual borrow bps on short weights charges positive funding drag."""
    w_prev = torch.tensor([0.5, 0.5, 0.0, 0.0])
    w = torch.tensor([1.0, 0.0, 0.0, -0.0])  # no short yet
    # Introduce a short leg of 0.25
    w = torch.tensor([0.75, 0.5, 0.0, -0.25])
    ret = torch.zeros(4)
    bd = apply_costs(
        w,
        w_prev,
        ret,
        arm=_eq_arm(),
        equity_bps=0.0,
        om_touch_enabled=False,
        borrow_bps_annual=100.0,
        borrow_dt_years=1.0 / 252.0,
    )
    # |w^-| = 0.25; cost = 0.25 * (100/1e4) * (1/252)
    expected = 0.25 * (100.0 / 1e4) * (1.0 / 252.0)
    assert bd.funding == pytest.approx(expected, abs=1e-9)
    assert bd.funding > 0.0


@pytest.mark.unit
def test_buy_then_sell_cost_symmetry():
    """Buy then sell the same amount: total equity spread equals 2x one-way cost."""
    w0 = torch.tensor([0.25, 0.25, 0.25, 0.25])
    w1 = torch.tensor([0.5, 0.5, 0.0, 0.0])
    ret = torch.zeros(4)
    bps = 10.0
    buy = apply_costs(
        w1, w0, ret, arm=_eq_arm(), equity_bps=bps, om_touch_enabled=False
    )
    sell = apply_costs(
        w0, w1, ret, arm=_eq_arm(), equity_bps=bps, om_touch_enabled=False
    )
    assert buy.equity_spread == pytest.approx(sell.equity_spread, **FLOAT_TOL)
    assert (buy.equity_spread + sell.equity_spread) == pytest.approx(
        2.0 * buy.equity_spread, **FLOAT_TOL
    )
