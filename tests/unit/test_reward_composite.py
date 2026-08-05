"""Srivastava composite reward wiring."""
from __future__ import annotations

import pytest

from mascotrl.policy.objective_factory import sdr_composite_reward
from tests.conftest import FLOAT_TOL


def test_sdr_composite_formula():
    w = {"w_ann": 1.0, "w_down": 1.0, "w_diff": 1.0, "w_treynor": 1.0}
    pnl = 0.01
    bench = 0.005
    out = sdr_composite_reward(pnl, bench_pnl=bench, beta=1.0, weights=w, ann_factor=252.0)
    expected = (
        1.0 * (0.01 * 252.0)
        - 1.0 * 0.0
        + 1.0 * (0.01 - 0.005)
        + 1.0 * (0.01 - 0.005) / 1.0
    )
    assert out == pytest.approx(expected, **FLOAT_TOL)


def test_sdr_composite_weights_honored():
    pnl = -0.02
    out = sdr_composite_reward(
        pnl, weights={"w_down": 2.0, "w_ann": 0.0, "w_diff": 0.0, "w_treynor": 0.0}
    )
    assert out == pytest.approx(-2.0 * 0.02, **FLOAT_TOL)
