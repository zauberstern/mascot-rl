"""Hedge impact charges through apply_costs when enabled + ADV set."""
from __future__ import annotations

import pytest
from tests.conftest import FLOAT_TOL
import numpy as np
import torch
from mascotrl.eval.friction import apply_costs

def test_hedge_impact_adds_to_hedge_leg_when_adv_set() -> None:
    w = torch.tensor([[0.5, -0.5]])
    w_prev = torch.zeros(1, 2)
    ret = torch.tensor([0.01, 0.01])
    base = apply_costs(w, w_prev, ret, hedge_leg_bps=0.0, om_touch_enabled=False, deltas=np.array([0.5, 0.5]), spot=np.array([100.0, 100.0]), capital_base=np.array([50.0, 50.0]), hedge_impact_enabled=True, hedge_impact_coef=1.0, hedge_adv=1000.0, hedge_sigma=0.2)
    zero_adv = apply_costs(w, w_prev, ret, hedge_leg_bps=0.0, om_touch_enabled=False, deltas=np.array([0.5, 0.5]), spot=np.array([100.0, 100.0]), capital_base=np.array([50.0, 50.0]), hedge_impact_enabled=True, hedge_impact_coef=1.0, hedge_adv=0.0, hedge_sigma=0.2)
    assert base.hedge_leg > 0.0
    assert zero_adv.hedge_leg == pytest.approx(0.0, **FLOAT_TOL)
    assert base.net < base.gross
