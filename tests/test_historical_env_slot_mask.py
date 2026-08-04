"""W4.2: HistoricalArmEnv slot-validity masking on rebalance days."""
from __future__ import annotations

import numpy as np
import pytest

from src.arms import ArmSpec
from src.env.historical_env import HistoricalArmEnv
from src.eval.friction import FrictionSpec
from src.eval.residualization import fit_ff4_residualizer, freeze_residualizer


def _env(rets, factors, *, slot_valid_mask=None, rebalance_mask=None):
    arm = ArmSpec(id="eq", option_slots=0, equity_slots=rets.shape[1], delta_mode="off")
    friction = FrictionSpec(om_touch_enabled=False, equity_bps=5.0)
    resid = freeze_residualizer(
        fit_ff4_residualizer(rets.mean(axis=1), factors, fold_id="p"), "p"
    )
    return HistoricalArmEnv(
        returns=rets,
        factors=factors,
        arm=arm,
        friction=friction,
        residualizer=resid,
        slot_valid_mask=slot_valid_mask,
        rebalance_mask=rebalance_mask,
    )


def test_slot_mask_zeros_inactive_slot_and_renormalizes_l1():
    rng = np.random.default_rng(0)
    t, k = 10, 4
    rets = rng.normal(scale=0.01, size=(t, k))
    factors = rng.normal(scale=0.005, size=(t, 4))
    mask = np.ones((t, k), dtype=bool)
    mask[:, 3] = False  # slot 3 permanently inactive

    env = _env(rets, factors, slot_valid_mask=mask)
    env.reset()
    w_raw = np.array([0.25, 0.25, 0.25, 0.25])
    _obs, _r, _term, _trunc, info = env.step(w_raw)

    post = info["post_fill_w"]
    assert post[3] == pytest.approx(0.0)
    # Original L1 budget (1.0) is preserved across the three active slots.
    assert float(np.abs(post).sum()) == pytest.approx(1.0, abs=1e-9)
    assert post[0] == pytest.approx(post[1]) == pytest.approx(post[2])


def test_slot_mask_all_inactive_keeps_zero_weights():
    rng = np.random.default_rng(1)
    t, k = 5, 3
    rets = rng.normal(scale=0.01, size=(t, k))
    factors = rng.normal(scale=0.005, size=(t, 4))
    mask = np.zeros((t, k), dtype=bool)

    env = _env(rets, factors, slot_valid_mask=mask)
    env.reset()
    _obs, _r, _term, _trunc, info = env.step(np.array([0.5, -0.3, 0.2]))
    assert np.allclose(info["post_fill_w"], 0.0)


def test_slot_mask_only_applied_on_rebalance_days_and_held_between():
    rng = np.random.default_rng(2)
    t, k = 6, 3
    rets = rng.normal(scale=0.01, size=(t, k))
    factors = rng.normal(scale=0.005, size=(t, 4))
    rebalance_mask = np.zeros(t, dtype=bool)
    rebalance_mask[[1, 4]] = True
    mask = np.ones((t, k), dtype=bool)
    mask[:, 2] = False

    env = _env(rets, factors, slot_valid_mask=mask, rebalance_mask=rebalance_mask)
    env.reset()
    # t=1 (rebalance): raw weights get masked+renormalized.
    _obs, _r, _term, _trunc, info1 = env.step(np.array([0.5, 0.5, 0.5]))
    assert info1["post_fill_w"][2] == pytest.approx(0.0)
    held = env.w.copy()
    # t=2..3 off-rebalance: weights held regardless of proposed weights.
    for _ in range(2):
        _obs, _r, _term, _trunc, info_off = env.step(np.array([9.0, -9.0, 9.0]))
        assert np.allclose(env.w, held)
        assert info_off["post_fill_w"][2] == pytest.approx(held[2])


def test_slot_handover_between_rebalances_produces_nonzero_turnover():
    """Changing which secid occupies a slot must be charged as turnover."""
    rng = np.random.default_rng(3)
    t, k = 4, 2
    rets = rng.normal(scale=0.01, size=(t, k))
    factors = rng.normal(scale=0.005, size=(t, 4))
    rebalance_mask = np.ones(t, dtype=bool)
    # Slot 1 flips from active (secid A) to inactive (secid B) across a
    # handover; a policy proposing non-equal weights across the handover
    # must see nonzero turnover from the mask-driven weight change alone.
    mask = np.array(
        [
            [True, True],
            [True, False],  # consumed by the first step() call (env.t == 1)
            [False, True],  # consumed by the second step() call: handover
            [True, True],
        ]
    )
    env = _env(rets, factors, slot_valid_mask=mask, rebalance_mask=rebalance_mask)
    env.reset()
    env.step(np.array([0.5, 0.5]))
    _obs, _r, _term, _trunc, info = env.step(np.array([0.5, 0.5]))
    assert info["turnover"] > 0.0


def test_slot_mask_shape_validation():
    rng = np.random.default_rng(4)
    rets = rng.normal(size=(5, 3))
    factors = rng.normal(size=(5, 4))
    bad_mask = np.ones((5, 2), dtype=bool)
    with pytest.raises(ValueError):
        _env(rets, factors, slot_valid_mask=bad_mask)
