"""TDD: transfer_report requires real reference arm for promotion."""
from __future__ import annotations

import math

import pytest

from src.eval.transfer_report import build_transfer_report, refuse_promotion_without_real_arm


def test_transfer_report_gap_fields() -> None:
    art = build_transfer_report(
        train_metric=1.0,
        eval_metric=0.7,
        train_world="gbm",
        eval_world="optionmetrics",
        real_reference_arm_present=True,
        real_reference_metric=0.8,
    )
    assert art["transfer_gap"] == pytest.approx(0.3)
    assert art["transfer_gap_pct"] == pytest.approx(0.3)
    assert art["metric_orientation"] == "higher_better"
    assert art["real_reference_arm_present"] is True
    assert art["train_world"] == "gbm"
    assert art["eval_world"] == "optionmetrics"


def test_transfer_report_lower_better_orientation() -> None:
    # Cost metric: train 0.5, eval 0.8 → train looks better (lower cost).
    # o=-1 * (0.5-0.8) = +0.3; positive gap means train flattered the policy.
    art = build_transfer_report(
        train_metric=0.5,
        eval_metric=0.8,
        train_world="heston",
        eval_world="optionmetrics",
        real_reference_arm_present=True,
        claim_metric="cao_y",
        metric_orientation="lower_better",
    )
    assert art["metric_orientation"] == "lower_better"
    assert art["transfer_gap"] == pytest.approx(0.3)
    assert art["transfer_gap_pct"] == pytest.approx(0.3 / 0.5)


def test_transfer_report_higher_better_explicit() -> None:
    art = build_transfer_report(
        train_metric=1.0,
        eval_metric=0.7,
        train_world="gbm",
        eval_world="optionmetrics",
        real_reference_arm_present=True,
        metric_orientation="higher_better",
    )
    assert art["transfer_gap"] == pytest.approx(0.3)


def test_transfer_report_nan_safe() -> None:
    art = build_transfer_report(
        train_metric=float("nan"),
        eval_metric=0.1,
        train_world="heston",
        eval_world="optionmetrics",
        real_reference_arm_present=False,
    )
    assert math.isnan(art["transfer_gap"])
    assert art["real_reference_arm_present"] is False


def test_refuse_promotion_without_real_arm() -> None:
    bad = build_transfer_report(
        train_metric=1.0,
        eval_metric=0.9,
        train_world="rbergomi",
        eval_world="optionmetrics",
        real_reference_arm_present=False,
    )
    with pytest.raises(ValueError, match="real_reference_arm"):
        refuse_promotion_without_real_arm(bad)


def test_allow_promotion_with_real_arm() -> None:
    good = build_transfer_report(
        train_metric=1.0,
        eval_metric=0.9,
        train_world="historical",
        eval_world="optionmetrics",
        real_reference_arm_present=True,
        real_reference_metric=0.95,
    )
    out = refuse_promotion_without_real_arm(good)
    assert out is good
