"""D4: equity fill ladder must vary equity_bps even when om_touch is on.

om_touch_spread_multiplier scales option drag only; on the eq arm it does
not change equity cost, so mid/pct75/worst collapse and gate1 break-even
becomes NaN. Ladder specs must stress equity_bps whenever eq_bps > 0.
"""
from __future__ import annotations

import math

import pytest

from mascotrl.eval.friction import FrictionSpec
from mascotrl.eval.research_alpha_cpcv import _fill_ladder_specs


def test_eq_arm_ladder_varies_equity_bps_even_with_om_touch() -> None:
    base = FrictionSpec(
        equity_bps=5.0,
        om_touch_enabled=True,
        om_touch_spread_multiplier=1.0,
    )
    specs = _fill_ladder_specs(base)
    assert specs["mid"].equity_bps == pytest.approx(2.5)
    assert specs["pct75"].equity_bps == pytest.approx(5.0)
    assert specs["worst"].equity_bps == pytest.approx(10.0)


def test_eq_arm_ladder_varies_equity_bps_when_om_touch_off() -> None:
    base = FrictionSpec(equity_bps=5.0, om_touch_enabled=False)
    specs = _fill_ladder_specs(base)
    assert specs["mid"].equity_bps == pytest.approx(2.5)
    assert specs["worst"].equity_bps == pytest.approx(10.0)


def test_zero_equity_bps_falls_back_to_om_touch_multiplier() -> None:
    base = FrictionSpec(
        equity_bps=0.0,
        om_touch_enabled=True,
        om_touch_spread_multiplier=1.0,
    )
    specs = _fill_ladder_specs(base)
    assert specs["mid"].om_touch_spread_multiplier == pytest.approx(0.5)
    assert specs["worst"].om_touch_spread_multiplier == pytest.approx(2.0)


def test_gate1_break_even_finite_when_ladder_varies() -> None:
    """mid > pct75 on a varied equity ladder => finite positive break-even."""
    from scripts.run_eq_alloc_campaign import compute_eq_campaign_gates

    # mid@0.5x Sharpe=1.024, pct75@1x Sharpe=1.0 -> slope negative, BE finite.
    gates = compute_eq_campaign_gates(
        fill_ladder={"mid": 1.024, "pct75": 1.0},
        policy_sharpe=1.0,
        challenger_sharpes={},
        series0=None,
        series0_dates=None,
        panel_dates=["2020-01-02", "2020-01-03"],
        lake_root="/nonexistent",
    )
    be = gates["gate1"]["break_even_spread_multiplier"]
    assert math.isfinite(be), f"expected finite break-even, got {be}"
    assert be > 0.0
