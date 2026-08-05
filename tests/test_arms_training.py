"""Symmetric opt/eq/mix arm training helpers."""
from __future__ import annotations

import numpy as np

from mascotrl.arms.spec import ArmSpec
from mascotrl.arms.training import (
    arm_training_manifest,
    build_arm_coverage,
    projection_backend_for_arm,
    resolve_portfolio_arm,
)


def test_resolve_eq_mix_opt():
    eq = resolve_portfolio_arm({"portfolio_arm": "eq", "n_assets": 10})
    assert eq.id == "eq" and eq.equity_slots == 10 and eq.option_slots == 0
    mix = resolve_portfolio_arm({"portfolio_arm": "mix", "n_assets": 10})
    assert mix.id == "mix" and mix.alpha_claim is False
    assert projection_backend_for_arm(mix) == "overlay_cvxpy"
    opt = resolve_portfolio_arm({"portfolio_arm": "opt", "n_assets": 10})
    assert opt.option_slots == 10
    assert projection_backend_for_arm(opt) == "cvxpy"


def test_mix_manifest_diagnostic_only():
    man = arm_training_manifest({"portfolio_arm": "mix", "n_assets": 8})
    assert man["diagnostic_only"] is True
    assert man["alpha_claim_allowed"] is False


def test_eq_coverage_no_atm_required():
    arm = ArmSpec(id="eq", option_slots=0, equity_slots=3, alpha_claim=True)
    T, K = 5, 3
    el = np.ones((T, K))
    el[2, 1] = np.nan
    mask = build_arm_coverage(arm, equity_labels=el)
    assert mask.shape == (T, K)
    assert mask[2, 1] == False  # noqa: E712
    assert mask[0, 0] == True  # noqa: E712
