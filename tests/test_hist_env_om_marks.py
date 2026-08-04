"""OM marks threading into HistoricalArmEnv for option friction."""
from __future__ import annotations

import numpy as np
import pytest

from src.arms.spec import ArmSpec
from src.env.historical_env import HistoricalArmEnv
from src.eval.friction import FrictionSpec


def _marks(T: int, K: int) -> dict[str, np.ndarray]:
    return {
        "half_spread": np.full((T, K), 0.25, dtype=np.float64),
        "capital_base": np.full((T, K), 50.0, dtype=np.float64),
        "delta": np.full((T, K), 0.5, dtype=np.float64),
        "spot": np.full((T, K), 100.0, dtype=np.float64),
    }


def test_opt_env_with_marks_charges_option_spread():
    T, K = 20, 3
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, size=(T, K))
    fac = rng.normal(0.0, 0.005, size=(T, 4))
    arm = ArmSpec(id="opt", option_slots=K, equity_slots=0)
    fr = FrictionSpec(
        equity_bps=0.0,
        hedge_leg_bps=5.0,
        om_touch_enabled=True,
        execution_spread_bps=5.0,
        borrow_floor_bps_annual=0.0,
    )
    env = HistoricalArmEnv(
        returns=rets,
        factors=fac,
        arm=arm,
        friction=fr,
        residualizer=None,
        marks=_marks(T, K),
        rebalance_mask=np.ones(T, dtype=bool),
    )
    env.reset(seed=0)
    w = np.array([0.5, 0.3, 0.2])
    _, _, _, _, info = env.step(w)
    assert info["cost"] > 0.0


def test_opt_env_without_marks_fail_closed():
    T, K = 10, 2
    rets = np.zeros((T, K))
    fac = np.zeros((T, 4))
    arm = ArmSpec(id="opt", option_slots=K, equity_slots=0)
    fr = FrictionSpec(om_touch_enabled=True, equity_bps=0.0)
    with pytest.raises(ValueError, match="om_touch|_om_marks|marks"):
        HistoricalArmEnv(
            returns=rets,
            factors=fac,
            arm=arm,
            friction=fr,
            residualizer=None,
            marks=None,
        )


def test_eq_env_without_marks_ok():
    T, K = 10, 2
    rets = np.zeros((T, K))
    fac = np.zeros((T, 4))
    arm = ArmSpec(id="eq", option_slots=0, equity_slots=K)
    fr = FrictionSpec(om_touch_enabled=True, equity_bps=5.0)
    env = HistoricalArmEnv(
        returns=rets,
        factors=fac,
        arm=arm,
        friction=fr,
        residualizer=None,
        marks=None,
        rebalance_mask=np.ones(T, dtype=bool),
    )
    env.reset()
    _, _, _, _, info = env.step(np.array([0.6, 0.4]))
    assert np.isfinite(info["cost"])
