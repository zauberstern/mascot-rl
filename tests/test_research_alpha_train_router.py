"""Research train router: primary_train selects HistoricalArmEnv + friction parity."""
from __future__ import annotations

import pytest

from mascotrl.eval.friction import FrictionSpec, assert_friction_parity, friction_spec_from_cfg
from mascotrl.reporting.research_alpha_router import (
    RESEARCH_PRIMARY_HIST,
    resolve_research_primary_train,
    research_train_friction_pair,
)


def test_resolve_historical_arm_env() -> None:
    out = resolve_research_primary_train({"primary_train": "historical_arm_env"})
    assert out == RESEARCH_PRIMARY_HIST


def test_resolve_unknown_primary_train_raises() -> None:
    with pytest.raises(ValueError, match="primary_train"):
        resolve_research_primary_train({"primary_train": "synth_cmdp_only"})


def test_resolve_missing_primary_defaults_refused() -> None:
    with pytest.raises(ValueError, match="primary_train"):
        resolve_research_primary_train({})


def test_research_friction_pair_parity_when_matched() -> None:
    cfg = {
        "om_touch_enabled": True,
        "om_touch_fee_bps": 0.0,
        "om_touch_spread_multiplier": 1.0,
        "hedge_leg_spread_bps": 5.0,
        "reward_shaping_ablation": True,
    }
    train, oos = research_train_friction_pair(cfg)
    assert isinstance(train, FrictionSpec)
    assert isinstance(oos, FrictionSpec)
    assert_friction_parity(train, oos)
    assert train.om_touch_enabled is True


def test_research_friction_pair_uses_friction_spec_from_cfg() -> None:
    cfg = {"om_touch_enabled": True, "hedge_leg_spread_bps": 5.0}
    train, oos = research_train_friction_pair(cfg)
    expected = friction_spec_from_cfg(cfg)
    assert train.hedge_leg_bps == expected.hedge_leg_bps
    assert oos.om_touch_enabled == expected.om_touch_enabled


def test_warmstart_zero_means_no_synth_primary() -> None:
    out = resolve_research_primary_train(
        {"primary_train": "historical_arm_env", "synth_warmstart_episodes": 0}
    )
    assert out == RESEARCH_PRIMARY_HIST
    meta = resolve_research_primary_train(
        {"primary_train": "historical_arm_env", "synth_warmstart_episodes": 0},
        with_meta=True,
    )
    assert meta["synth_warmstart_episodes"] == 0
    assert meta["synth_primary"] is False
