"""Step 13: reward = gross - cost - factor at machine precision."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.arms import ArmSpec
from src.env.historical_env import HistoricalArmEnv
from src.eval.friction import FrictionSpec, apply_costs
from src.eval.residualization import (
    fit_ff4_residualizer,
    freeze_residualizer,
    residualize_step,
)


def test_reward_parity_machine_precision():
    rng = np.random.default_rng(7)
    T, K = 30, 4
    rets = rng.normal(scale=0.01, size=(T, K))
    factors = rng.normal(scale=0.005, size=(T, 4))
    arm = ArmSpec(id="eq", option_slots=0, equity_slots=K, delta_mode="off")
    friction = FrictionSpec(
        spec_id="v2_quote_touch",
        equity_bps=5.0,
        om_touch_enabled=False,
        cost_multiplier=1.0,
        borrow_floor_bps_annual=0.0,
    )
    resid = freeze_residualizer(
        fit_ff4_residualizer(rets.mean(axis=1), factors, fold_id="parity"),
        "parity",
    )
    env = HistoricalArmEnv(
        returns=rets,
        factors=factors,
        arm=arm,
        friction=friction,
        residualizer=resid,
        rf=np.zeros(T),
        portfolio_beta_window=0,  # use frozen residualizer betas
    )
    env.reset()

    # RC5: env cold-starts at EW; mirror that for the hand-rolled cost check.
    w_prev = np.full(K, 1.0 / K, dtype=np.float64)
    for t in range(1, T):
        w = rng.normal(size=K)
        w = w - w.mean()
        g = np.abs(w).sum()
        if g > 0:
            w = w / g

        _obs, reward, terminated, _trunc, info = env.step(w)

        ret_t = rets[env.t - 1]  # step advanced; last consumed row
        # Recompute expected using the same primitives as the env contract.
        breakdown = apply_costs(
            torch.as_tensor(w, dtype=torch.float64),
            torch.as_tensor(w_prev, dtype=torch.float64),
            torch.as_tensor(ret_t, dtype=torch.float64),
            arm=arm,
            friction=friction,
        )
        cost = float(
            breakdown.option_spread
            + breakdown.equity_spread
            + breakdown.hedge_leg
            + breakdown.funding
        )
        fac_t = factors[env.t - 1]
        expected = residualize_step(
            breakdown.gross,
            cost,
            resid.betas,
            fac_t,
            borrow=0.0,
            rf=0.0,
        )
        assert info["gross"] == pytest.approx(breakdown.gross, rel=0, abs=1e-15)
        assert info["cost"] == pytest.approx(cost, rel=0, abs=1e-15)
        assert float(reward) == pytest.approx(expected, rel=0, abs=1e-15)
        assert info["residual"] == pytest.approx(expected, rel=0, abs=1e-15)
        assert info["residual"] == pytest.approx(
            info["gross"]
            - info["cost"]
            - info["borrow"]
            - info["rf"]
            - info["factor"],
            rel=0,
            abs=1e-15,
        )
        w_prev = w.copy()
        if terminated:
            break
