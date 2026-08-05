"""Unified friction: HAPPO and baselines share the same cost function."""
from __future__ import annotations

import numpy as np
import pytest
from tests.conftest import FLOAT_TOL
import torch

from mascotrl.arms import ArmSpec
from mascotrl.eval.friction import FrictionBreakdown, FrictionSpec, apply_costs


def test_status_quo_arm_none_produces_finite_net():
    w = torch.tensor([[0.4, -0.2]])
    w_prev = torch.zeros(1, 2)
    ret = torch.tensor([0.01, -0.005])
    hs = torch.tensor([[0.10, 0.05]])
    cb = torch.tensor([[40.0, 40.0]])
    out = apply_costs(
        w,
        w_prev,
        ret,
        arm=None,
        half_spread=hs,
        capital_base=cb,
        om_touch_enabled=True,
        spread_multiplier=1.0,
        hedge_leg_bps=0.0,
    )
    assert isinstance(out, FrictionBreakdown)
    assert np.isfinite(out.gross)
    assert np.isfinite(out.option_spread)
    assert out.equity_spread == pytest.approx(0.0, **FLOAT_TOL)
    assert np.isfinite(out.net)
    assert out.net == out.gross - out.option_spread - out.equity_spread - out.hedge_leg - out.funding


def test_equity_only_arm_charges_equity_bps_not_option_spread():
    arm = ArmSpec(id="eq", option_slots=0, equity_slots=2, delta_mode="off")
    w = torch.tensor([[0.5, -0.5]])
    w_prev = torch.zeros(1, 2)
    ret = torch.tensor([0.01, 0.02])
    # half_spread present but unused for equity-only when OM-touch is off
    hs = torch.tensor([[9.0, 9.0]])
    out = apply_costs(
        w,
        w_prev,
        ret,
        arm=arm,
        half_spread=hs,
        om_touch_enabled=False,
        equity_bps=5.0,
        spread_multiplier=1.0,
        hedge_leg_bps=5.0,
        deltas=np.array([1.0, 1.0]),
        spot=np.array([100.0, 100.0]),
        capital_base=np.array([100.0, 100.0]),
    )
    turn = float((w - w_prev).abs().sum().item())  # 1.0
    expected_eq = (5.0 / 1e4) * 1.0 * turn
    assert out.option_spread == pytest.approx(0.0, **FLOAT_TOL)
    assert out.hedge_leg == pytest.approx(0.0, **FLOAT_TOL)
    assert out.equity_spread == pytest.approx(expected_eq)
    assert out.net < out.gross


def test_mix_arm_splits_costs_across_blocks():
    arm = ArmSpec(id="mix", option_slots=2, equity_slots=2, delta_mode="joint")
    w = torch.tensor([[0.3, 0.0, 0.4, -0.2]])
    w_prev = torch.zeros(1, 4)
    ret = torch.tensor([0.01, 0.0, 0.02, -0.01])
    hs = torch.tensor([[0.20, 0.10, 99.0, 99.0]])  # equity half-spreads ignored
    cb = torch.tensor([[50.0, 50.0, 1.0, 1.0]])
    out = apply_costs(
        w,
        w_prev,
        ret,
        arm=arm,
        half_spread=hs,
        capital_base=cb,
        om_touch_enabled=True,
        spread_multiplier=1.0,
        equity_bps=5.0,
        hedge_leg_bps=0.0,
    )
    assert out.option_spread > 0.0
    assert out.equity_spread > 0.0
    # Equity turn = |0.4| + |-0.2| = 0.6
    assert out.equity_spread == pytest.approx((5.0 / 1e4) * 0.6)
    # Option touch uses only first two slots: |0.3|*0.20/50 + 0 = 0.0012
    assert out.option_spread == pytest.approx(0.0012)
    assert out.net == out.gross - out.option_spread - out.equity_spread - out.hedge_leg - out.funding


def test_spread_multiplier_ladder_weakens_net_when_option_costs_positive():
    arm = ArmSpec(id="opt", option_slots=2, equity_slots=0, delta_mode="soft")
    w = torch.tensor([[0.5, -0.5]])
    w_prev = torch.zeros(1, 2)
    ret = torch.tensor([0.01, 0.01])
    hs = torch.tensor([[0.25, 0.25]])
    cb = torch.tensor([[50.0, 50.0]])
    nets = []
    for m in (0.0, 0.25, 0.50, 1.0):
        out = apply_costs(
            w,
            w_prev,
            ret,
            arm=arm,
            half_spread=hs,
            capital_base=cb,
            om_touch_enabled=True,
            spread_multiplier=m,
            hedge_leg_bps=0.0,
        )
        assert out.option_spread >= 0.0
        nets.append(out.net)
    # Higher multiplier → lower (or equal) net
    for a, b in zip(nets, nets[1:]):
        assert b <= a + 1e-12
    assert nets[0] > nets[-1]


def test_nan_label_does_not_poison_gross():
    """One NaN label with positive weight: gross = sum over finite slots only."""
    w = torch.tensor([0.5, 0.5])
    w_prev = torch.zeros(2)
    ret = torch.tensor([0.02, float("nan")])
    out = apply_costs(w, w_prev, ret, arm=None, om_touch_enabled=False)
    assert out.gross == pytest.approx(0.01)
    assert out.n_nan_labels == 1


def test_all_nan_row_gross_zero():
    w = torch.tensor([0.4, 0.6])
    w_prev = torch.zeros(2)
    ret = torch.tensor([float("nan"), float("nan")])
    out = apply_costs(w, w_prev, ret, arm=None, om_touch_enabled=False)
    assert out.gross == pytest.approx(0.0, **FLOAT_TOL)
    assert out.n_nan_labels == 2


def test_equity_spread_multiplier_independent_of_om_touch():
    arm = ArmSpec(id="eq", option_slots=0, equity_slots=2, delta_mode="off")
    w = torch.tensor([[0.5, -0.5]])
    w_prev = torch.zeros(1, 2)
    ret = torch.tensor([0.01, 0.02])
    base = FrictionSpec(
        equity_bps=10.0,
        om_touch_enabled=False,
        om_touch_spread_multiplier=2.0,
        equity_spread_multiplier=1.0,
        impact_c_eq=0.0,
        execution_impact_coef=0.0,
    )
    out_base = apply_costs(w, w_prev, ret, arm=arm, friction=base)
    out_om2 = apply_costs(
        w,
        w_prev,
        ret,
        arm=arm,
        friction=FrictionSpec(
            equity_bps=10.0,
            om_touch_enabled=False,
            om_touch_spread_multiplier=2.0,
            equity_spread_multiplier=1.0,
            impact_c_eq=0.0,
            execution_impact_coef=0.0,
        ),
    )
    out_eq2 = apply_costs(
        w,
        w_prev,
        ret,
        arm=arm,
        friction=FrictionSpec(
            equity_bps=10.0,
            om_touch_enabled=False,
            om_touch_spread_multiplier=1.0,
            equity_spread_multiplier=2.0,
            impact_c_eq=0.0,
            execution_impact_coef=0.0,
        ),
    )
    assert out_base.equity_spread == pytest.approx(out_om2.equity_spread)
    assert out_eq2.equity_spread == pytest.approx(2.0 * out_base.equity_spread)
