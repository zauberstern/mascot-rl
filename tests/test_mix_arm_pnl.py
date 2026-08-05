"""Mix arm: both equity_spread and option_spread fire on concatenated panel."""
from __future__ import annotations

import numpy as np

from mascotrl.arms.spec import ArmSpec
from mascotrl.env.historical_env import HistoricalArmEnv
from mascotrl.eval.friction import FrictionSpec


def test_mix_env_charges_both_blocks():
    T, n_opt, n_eq = 15, 2, 2
    K = n_opt + n_eq
    rng = np.random.default_rng(1)
    rets = rng.normal(0.001, 0.01, size=(T, K))
    fac = rng.normal(0.0, 0.005, size=(T, 4))
    arm = ArmSpec(id="mix", option_slots=n_opt, equity_slots=n_eq, alpha_claim=False)
    fr = FrictionSpec(
        equity_bps=5.0,
        hedge_leg_bps=5.0,
        om_touch_enabled=True,
        execution_spread_bps=5.0,
        borrow_floor_bps_annual=0.0,
        impact_c_eq=0.0,
        execution_impact_coef=0.0,
    )
    marks = {
        "half_spread": np.concatenate(
            [np.full((T, n_opt), 0.25), np.zeros((T, n_eq))], axis=1
        ),
        "capital_base": np.concatenate(
            [np.full((T, n_opt), 50.0), np.zeros((T, n_eq))], axis=1
        ),
        "delta": np.concatenate(
            [np.full((T, n_opt), 0.5), np.zeros((T, n_eq))], axis=1
        ),
        "spot": np.full((T, K), 100.0),
    }
    env = HistoricalArmEnv(
        returns=rets,
        factors=fac,
        arm=arm,
        friction=fr,
        residualizer=None,
        marks=marks,
        rebalance_mask=np.ones(T, dtype=bool),
    )
    env.reset(seed=0)
    # Nonzero turnover on both blocks.
    w = np.array([0.3, 0.2, 0.3, 0.2])
    _, _, _, _, info = env.step(w)
    assert info["cost"] > 0.0
    # Direct apply_costs breakdown via a second step from nonzero w_prev.
    from mascotrl.eval.friction import apply_costs
    import torch

    out = apply_costs(
        torch.as_tensor(w),
        torch.zeros(K),
        torch.as_tensor(rets[1]),
        arm=arm,
        friction=fr,
        half_spread=marks["half_spread"][1],
        capital_base=marks["capital_base"][1],
        deltas=marks["delta"][1],
        deltas_prev=marks["delta"][0],
        spot=marks["spot"][1],
    )
    assert out.equity_spread > 0.0
    assert out.option_spread > 0.0
