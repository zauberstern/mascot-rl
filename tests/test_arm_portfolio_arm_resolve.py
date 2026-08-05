"""WP-S1: portfolio_arm must resolve eq/mix instead of silently defaulting to opt."""
from __future__ import annotations

from mascotrl.arms.spec import arm_spec_from_cfg


def test_portfolio_arm_eq_resolves() -> None:
    arm = arm_spec_from_cfg({"portfolio_arm": "eq", "n_assets": 100})
    assert arm.id == "eq"
    assert arm.equity_slots == 100
    assert arm.option_slots == 0


def test_portfolio_arm_mix_resolves() -> None:
    arm = arm_spec_from_cfg({"portfolio_arm": "mix", "n_assets": 50})
    assert arm.id == "mix"
    assert arm.option_slots + arm.equity_slots == 50
    assert arm.alpha_claim is False


def test_absent_arm_and_portfolio_defaults_opt() -> None:
    arm = arm_spec_from_cfg({"n_assets": 40})
    assert arm.id == "opt"
    assert arm.option_slots == 40


def test_explicit_arm_block_wins() -> None:
    arm = arm_spec_from_cfg(
        {
            "portfolio_arm": "eq",
            "n_assets": 10,
            "arm": {"id": "opt", "option_slots": 10, "equity_slots": 0},
        }
    )
    assert arm.id == "opt"
