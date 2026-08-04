"""Part D.3: mtm_pnl reward is gross MTM minus costs, not residual."""
from __future__ import annotations

import numpy as np
import pytest

from src.arms import ArmSpec
from src.env.historical_env import HistoricalArmEnv
from src.eval.friction import FrictionSpec
from src.eval.residualization import fit_ff4_residualizer, freeze_residualizer


def test_mtm_pnl_reward_equals_gross_minus_costs_not_residual() -> None:
    rng = np.random.default_rng(0)
    t, k = 30, 4
    rets = rng.normal(0.001, 0.01, size=(t, k))
    factors = rng.normal(0.0, 0.005, size=(t, 4))
    arm = ArmSpec(id="eq", option_slots=0, equity_slots=k, delta_mode="off")
    friction = FrictionSpec(
        spec_id="v2_quote_touch", equity_bps=10.0, om_touch_enabled=False
    )
    resid = freeze_residualizer(
        fit_ff4_residualizer(rets.mean(axis=1), factors, fold_id="mtm"), "mtm"
    )
    env_res = HistoricalArmEnv(
        returns=rets,
        factors=factors,
        arm=arm,
        friction=friction,
        residualizer=resid,
        reward_mode="residual",
    )
    env_mtm = HistoricalArmEnv(
        returns=rets,
        factors=factors,
        arm=arm,
        friction=friction,
        residualizer=resid,
        reward_mode="mtm_pnl",
    )
    w = np.full(k, 1.0 / k, dtype=np.float64)
    env_res.reset(seed=0)
    env_mtm.reset(seed=0)
    _, r_res, _, _, info_res = env_res.step(w)
    _, r_mtm, _, _, info_mtm = env_mtm.step(w)
    expected = float(info_mtm["gross"]) - float(info_mtm["cost"]) - float(
        info_mtm["borrow"]
    )
    assert r_mtm == pytest.approx(expected, rel=0, abs=1e-12)
    assert info_mtm["mtm_pnl"] == pytest.approx(expected, rel=0, abs=1e-12)
    assert info_mtm["reward_mode"] == "mtm_pnl"
    # Residual path unchanged.
    assert r_res == pytest.approx(float(info_res["residual"]), rel=0, abs=1e-12)
    # mtm_pnl must differ from residual whenever factor/borrow/rf drag is nonzero.
    assert abs(float(info_mtm["residual"]) - expected) > 1e-15 or abs(
        float(info_mtm["factor"])
    ) + abs(float(info_mtm["borrow"])) + abs(float(info_mtm["rf"])) < 1e-15
    assert r_mtm != pytest.approx(float(info_mtm["residual"])) or abs(
        float(info_mtm["factor"])
    ) < 1e-18
