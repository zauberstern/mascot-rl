"""Step 11: HistoricalArmEnv smoke on numpy hist panels (no rBergomi)."""
from __future__ import annotations

import numpy as np
import pytest

from src.arms import ArmSpec
from src.env.historical_env import HistoricalArmEnv
from src.eval.friction import FrictionSpec
from src.eval.residualization import fit_ff4_residualizer, freeze_residualizer


def _toy_panel(T: int = 40, K: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(scale=0.01, size=(T, K))
    factors = rng.normal(scale=0.005, size=(T, 4))
    return rets, factors


def test_hist_env_reset_step_info_keys():
    rets, factors = _toy_panel()
    arm = ArmSpec(id="eq", option_slots=0, equity_slots=5, delta_mode="off")
    friction = FrictionSpec(spec_id="v2_quote_touch", equity_bps=5.0, om_touch_enabled=False)
    y = rets.mean(axis=1)
    resid = freeze_residualizer(fit_ff4_residualizer(y, factors, fold_id="f0"), "f0")

    env = HistoricalArmEnv(
        returns=rets,
        factors=factors,
        arm=arm,
        friction=friction,
        residualizer=resid,
    )
    obs, info0 = env.reset()
    assert obs.shape[-1] == arm.n_slots
    assert isinstance(info0, dict)

    w = np.zeros(arm.n_slots, dtype=np.float64)
    w[0], w[1] = 0.5, -0.5
    obs2, reward, terminated, truncated, info = env.step(w)
    assert np.isfinite(reward)
    for key in ("gross", "cost", "factor", "residual"):
        assert key in info
        assert np.isfinite(info[key])
    assert info["residual"] == pytest.approx(float(reward))
    assert not (terminated and truncated)


def test_hist_env_rejects_surface_tensors():
    """Constructor must not accept rBergomi surface tensors as the return panel."""
    arm = ArmSpec(id="eq", option_slots=0, equity_slots=2, delta_mode="off")
    friction = FrictionSpec(om_touch_enabled=False)
    fake_surface = np.zeros((4, 2, 10, 3, 3))  # rBergomi-like rank-5
    with pytest.raises((ValueError, TypeError)):
        HistoricalArmEnv(
            returns=fake_surface,
            factors=np.zeros((10, 4)),
            arm=arm,
            friction=friction,
            residualizer=None,
        )
