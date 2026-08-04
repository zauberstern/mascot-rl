"""Unit tests for slot masking (Phase E)."""
from __future__ import annotations

import numpy as np
import pytest
from tests.conftest import FLOAT_TOL

from src.data.slot_mask import (
    apply_slot_mask,
    masked_pnl,
    select_slots_for_date,
    valid_mask_from_slots,
)


def test_select_slots_pads_and_truncates():
    slots = select_slots_for_date([10, 20], k=4)
    assert slots == [10, 20, None, None]
    assert select_slots_for_date([1, 2, 3, 4, 5], k=3) == [1, 2, 3]


def test_mask_zeros_inactive_weights_and_pnl():
    slots = select_slots_for_date([1, 2], k=3)
    mask = valid_mask_from_slots(slots)
    w = np.array([0.5, -0.5, 0.9])
    w_m = apply_slot_mask(w, mask)
    assert w_m[2] == pytest.approx(0.0, **FLOAT_TOL)
    labels = np.array([0.01, -0.02, 9.0])
    pnl = masked_pnl(w, labels, mask)
    assert pnl == pytest.approx(0.5 * 0.01 + (-0.5) * (-0.02))


def test_future_eligibility_shuffle_does_not_change_past_masks():
    from src.data.slot_mask import build_slot_masks_over_dates

    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    eligible = {
        "2020-01-02": [1, 2, 3],
        "2020-01-03": [1, 2],
        "2020-01-06": [4, 5, 6, 7],
    }
    _, m_full = build_slot_masks_over_dates(dates, eligible, k=3)
    _, m_trunc = build_slot_masks_over_dates(dates[:2], eligible, k=3)
    np.testing.assert_array_equal(m_full[:2], m_trunc)
    eligible_future = dict(eligible)
    eligible_future["2020-01-06"] = [99, 98, 97]
    _, m_shuf = build_slot_masks_over_dates(dates, eligible_future, k=3)
    np.testing.assert_array_equal(m_full[:2], m_shuf[:2])


def test_coverage_masks_require_finite_atm_and_label():
    from src.data.slot_mask import coverage_masks_from_features

    atm = np.array([[0.2, np.nan], [0.1, 0.3]])
    lab = np.array([[0.01, 0.02], [np.nan, 0.03]])
    m = coverage_masks_from_features(atm, lab)
    assert m.tolist() == [[True, False], [False, True]]


def test_membership_masks_gate_non_members():
    from src.data.slot_mask import membership_masks_for_fixed_slots

    dates = ["2020-01-02", "2020-01-03"]
    secids = [1, 2, 3]
    tickers = ["AAA", "BBB", "CCC"]
    members = {
        "2020-01-02": ["AAA", "BBB"],
        "2020-01-03": ["CCC"],
    }
    m = membership_masks_for_fixed_slots(
        dates, secids, tickers=tickers, members_by_date=members
    )
    assert m.tolist() == [[True, True, False], [False, False, True]]


def test_membership_intersect_coverage_for_equity_arm():
    """Name occupies a slot only while an index member (coverage ∩ membership)."""
    from src.data.slot_mask import membership_masks_for_fixed_slots

    dates = ["2020-01-02", "2020-01-03"]
    secids = [10, 20]
    tickers = ["AAA", "BBB"]
    members = {
        "2020-01-02": ["AAA", "BBB"],
        "2020-01-03": ["AAA"],
    }
    coverage = np.array([[True, True], [True, True]], dtype=bool)
    m = membership_masks_for_fixed_slots(
        dates,
        secids,
        tickers=tickers,
        members_by_date=members,
        coverage=coverage,
    )
    assert m.tolist() == [[True, True], [True, False]]
