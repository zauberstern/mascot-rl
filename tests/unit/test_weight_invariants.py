"""Structural weight-head invariants that must hold for all valid inputs."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mascotrl.policy.cmdp_projector import turnover_cap_project
from mascotrl.policy.single_agent import _apply_weight_head
from tests.conftest import FLOAT_TOL

_LONG_ONLY = ("softmax", "sparse_tilt", "sparse_tilt_tsallis", "entmax_15", "dirichlet_tilt")


@pytest.mark.unit
@pytest.mark.parametrize("head", list(_LONG_ONLY))
def test_weight_head_simplex(head, rng):
    """Long-only heads produce non-negative weights that sum to 1."""
    for _ in range(50):
        raw = torch.as_tensor(rng.normal(size=(1, 10)), dtype=torch.float32)
        if head == "dirichlet_tilt":
            u = torch.softmax(raw, dim=-1)
            w = _apply_weight_head(
                u, head, tilt_gain=5.0, w_base=torch.full((10,), 0.1)
            )
        else:
            w = _apply_weight_head(
                raw, head, tilt_gain=5.0, w_base=torch.full((10,), 0.1)
            )
        assert w.shape == (1, 10)
        assert bool((w >= -1e-6).all()), f"negative weight under {head}: {w}"
        assert float(w.sum()) == pytest.approx(1.0, **FLOAT_TOL)


@pytest.mark.unit
def test_tanh_l1_unit_norm(rng):
    """tanh_l1 is long-short with unit L1, not a long-only simplex."""
    for _ in range(50):
        raw = torch.as_tensor(rng.normal(size=(1, 10)), dtype=torch.float32)
        w = _apply_weight_head(raw, "tanh_l1")
        assert w.shape == (1, 10)
        assert float(w.abs().sum()) == pytest.approx(1.0, **FLOAT_TOL)


@pytest.mark.unit
def test_turnover_non_negative(rng):
    """Turnover after projection must be within the cap and non-negative."""
    for _ in range(30):
        w_prev = np.asarray(
            torch.softmax(
                torch.as_tensor(rng.normal(size=(10,)), dtype=torch.float32), dim=-1
            ),
            dtype=np.float64,
        )
        w_raw = np.asarray(
            torch.softmax(
                torch.as_tensor(rng.normal(size=(10,)), dtype=torch.float32), dim=-1
            ),
            dtype=np.float64,
        )
        cap = 0.15
        w_proj = turnover_cap_project(w_raw, w_prev=w_prev, tau=cap)
        turnover = float(np.abs(w_proj - w_prev).sum())
        assert turnover >= -1e-9
        assert turnover <= cap + 1e-5


@pytest.mark.unit
def test_friction_cost_non_negative(rng):
    """Equity spread cost on a weight transition must be non-negative."""
    from mascotrl.arms import ArmSpec
    from mascotrl.eval.friction import apply_costs

    arm = ArmSpec(id="eq", option_slots=0, equity_slots=5, delta_mode="off")
    for _ in range(30):
        w_prev = torch.softmax(
            torch.as_tensor(rng.normal(size=(5,)), dtype=torch.float32), dim=-1
        )
        w = torch.softmax(
            torch.as_tensor(rng.normal(size=(5,)), dtype=torch.float32), dim=-1
        )
        ret = torch.as_tensor(rng.normal(size=(5,), scale=0.01), dtype=torch.float32)
        bd = apply_costs(
            w, w_prev, ret, arm=arm, equity_bps=5.0, om_touch_enabled=False
        )
        cost = float(bd.equity_spread + bd.option_spread + bd.hedge_leg + bd.funding)
        assert cost >= -1e-12
