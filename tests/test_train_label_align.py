"""Train/claim label stem alignment + eq historical_arm_env route helper."""
from __future__ import annotations

import pytest

from mascotrl.reporting.research_alpha_router import (
    assert_train_claim_label_align,
    should_route_historical_arm_env,
)


def test_assert_train_claim_label_align_ok():
    assert_train_claim_label_align(
        {"train_label_stem": "stk_ret", "claim_label_stem": "stk_ret"}
    )


def test_assert_train_claim_label_align_mismatch_raises():
    with pytest.raises(AssertionError, match="train_label_stem"):
        assert_train_claim_label_align(
            {"train_label_stem": "stk_ret", "claim_label_stem": "dh_ret_lagdelta"}
        )


def test_assert_train_claim_skips_when_missing():
    assert_train_claim_label_align({"claim_label_stem": "stk_ret"})
    assert_train_claim_label_align({})


def test_should_route_eq_historical_arm_env():
    assert should_route_historical_arm_env(
        {"primary_train": "historical_arm_env", "arm": {"id": "eq"}}
    )
    assert not should_route_historical_arm_env(
        {"primary_train": "historical_arm_env", "arm": {"id": "opt"}}
    )
    assert should_route_historical_arm_env(
        {"train_world": "historical", "portfolio_arm": "eq"}
    )
