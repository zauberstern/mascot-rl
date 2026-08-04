"""Arm-aware coverage masks (equity slots without ATM IV)."""
from __future__ import annotations

import numpy as np

from src.arms import ArmSpec
from src.data.slot_mask import coverage_masks_for_arm, coverage_masks_from_features


def test_opt_coverage_matches_legacy():
    atm = np.array([[0.2, np.nan], [0.1, 0.3]])
    lab = np.array([[0.01, 0.02], [np.nan, 0.03]])
    legacy = coverage_masks_from_features(atm, lab)
    arm = ArmSpec(id="opt", option_slots=2, equity_slots=0, delta_mode="soft")
    m = coverage_masks_for_arm(
        arm=arm,
        atm=atm,
        option_labels=lab,
    )
    np.testing.assert_array_equal(m, legacy)


def test_equity_coverage_without_atm_iv():
    """Equity slots are active when the equity label is finite; ATM IV not required."""
    eq_lab = np.array(
        [
            [0.01, np.nan],
            [np.nan, 0.02],
            [0.0, -0.01],
        ],
        dtype=np.float64,
    )
    # ATM all-NaN must not kill equity coverage.
    atm = np.full_like(eq_lab, np.nan)
    arm = ArmSpec(id="eq", option_slots=0, equity_slots=2, delta_mode="off")
    m = coverage_masks_for_arm(arm=arm, equity_labels=eq_lab, atm=atm)
    assert m.tolist() == [[True, False], [False, True], [True, True]]


def test_equity_coverage_optional_spot_gate():
    eq_lab = np.array([[0.01, 0.02]], dtype=np.float64)
    spot = np.array([[100.0, np.nan]], dtype=np.float64)
    arm = ArmSpec(id="eq", option_slots=0, equity_slots=2, delta_mode="off")
    m = coverage_masks_for_arm(arm=arm, equity_labels=eq_lab, spot=spot)
    assert m.tolist() == [[True, False]]


def test_mix_block_specific_coverage():
    atm = np.array([[0.2, np.nan]], dtype=np.float64)
    opt_lab = np.array([[0.01, 0.02]], dtype=np.float64)
    eq_lab = np.array([[0.03, np.nan]], dtype=np.float64)
    arm = ArmSpec(id="mix", option_slots=2, equity_slots=2, delta_mode="joint")
    m = coverage_masks_for_arm(
        arm=arm,
        atm=atm,
        option_labels=opt_lab,
        equity_labels=eq_lab,
    )
    assert m.shape == (1, 4)
    # Option: need atm AND label → [True, False]
    # Equity: label only → [True, False]
    assert m.tolist() == [[True, False, True, False]]
