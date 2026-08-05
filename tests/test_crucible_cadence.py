"""CRUCIBLE cadence: quarterly_63d universe mask vs policy mask."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.conftest import FLOAT_TOL

from mascotrl.eval.cadence import (
    assert_universe_subset_of_policy,
    build_rebalance_mask,
    build_universe_cadence_mask,
    quarterly_63d_mask,
)


def _dates(n: int = 200) -> pd.DatetimeIndex:
    return pd.bdate_range("2018-01-02", periods=n)


def test_quarterly_63d_fires_on_trading_day_indices():
    dates = _dates(200)
    m = quarterly_63d_mask(dates, every=63, anchor_index=0)
    assert m.dtype == bool
    assert list(np.flatnonzero(m)) == [0, 63, 126, 189]


def test_build_universe_cadence_quarterly():
    dates = _dates(130)
    m = build_universe_cadence_mask(dates, "quarterly_63d")
    assert list(np.flatnonzero(m)) == [0, 63, 126]


def test_build_rebalance_mask_accepts_quarterly_63d():
    dates = _dates(70)
    m = build_rebalance_mask(dates, "quarterly_63d")
    assert m is not None
    assert m[0] and m[63] and not m[1]


def test_universe_mask_subset_of_daily_policy():
    dates = _dates(100)
    u = build_universe_cadence_mask(dates, "quarterly_63d")
    # daily policy => None => always ok
    assert_universe_subset_of_policy(u, None)


def test_universe_mask_subset_of_weekly_policy_fails_when_misaligned():
    dates = _dates(100)
    u = build_universe_cadence_mask(dates, "quarterly_63d")
    # Force a universe-true day that is policy-false
    p = np.zeros(len(dates), dtype=bool)
    p[1] = True
    with pytest.raises(AssertionError):
        assert_universe_subset_of_policy(u, p)


def test_unknown_universe_cadence_raises():
    with pytest.raises(ValueError, match="unknown universe cadence"):
        build_universe_cadence_mask(_dates(10), "yearly")


def test_reselect_churn_never_exceeds_cap():
    from mascotrl.data.crucible import apply_reselect_churn_cap

    incumbent = list(range(100))
    proposed = list(range(50, 150))  # 50% churn if taken raw
    out = apply_reselect_churn_cap(incumbent, proposed, cap=0.25)
    changed = len(set(out) - set(incumbent))
    assert changed <= 25
    assert len(out) == 100


def test_incumbents_win_ties_hysteresis():
    from mascotrl.data.crucible import apply_reselect_churn_cap

    incumbent = [1, 2, 3, 4]
    proposed = [1, 2, 5, 6]
    ranks = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}
    out = apply_reselect_churn_cap(
        incumbent, proposed, cap=0.25, ranks=ranks, prefer_incumbent_on_tie=True
    )
    assert 1 in out and 2 in out
    assert len(set(out) - set(incumbent)) <= 1


def test_selection_and_policy_turnover_are_separate_keys():
    from mascotrl.data.crucible import separate_turnover_keys

    diag = separate_turnover_keys(selection_turnover=0.12, policy_turnover=0.45)
    assert "selection_turnover" in diag
    assert "policy_turnover" in diag
    assert diag["selection_turnover"] == pytest.approx(0.12, **FLOAT_TOL)
    assert diag["policy_turnover"] == pytest.approx(0.45, **FLOAT_TOL)
    assert "combined_turnover" not in diag
