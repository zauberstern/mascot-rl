"""Phase B: monthly / weekly rebalance cadence gates."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.conftest import FLOAT_TOL

from src.arms import ArmSpec
from src.eval.cadence import (
    annualized_turnover,
    build_rebalance_mask,
    month_end_mask,
    week_end_mask,
)
from src.eval.friction import FrictionSpec
from src.eval.parity_harness import score_equal_weight
from src.eval.residualization import fit_ff4_residualizer, freeze_residualizer


def _dates(n: int = 252, start: str = "2015-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def test_month_end_mask_marks_last_trading_day_of_month():
    dates = _dates(60)
    mask = month_end_mask(dates)
    assert mask.dtype == bool
    assert mask.shape == (len(dates),)
    # At least one month-end in 60 business days.
    assert int(mask.sum()) >= 2
    # Off days are False.
    assert bool(mask.sum()) < len(dates)
    # Last day of each month group is True.
    s = pd.Series(np.arange(len(dates)), index=dates)
    expected = s.groupby(s.index.to_period("M")).transform("max") == s
    np.testing.assert_array_equal(mask, expected.to_numpy())


def test_week_end_mask_marks_last_trading_day_of_week():
    dates = _dates(40)
    mask = week_end_mask(dates)
    assert int(mask.sum()) >= 5
    assert int(mask.sum()) < len(dates)


def test_build_rebalance_mask_dispatch():
    dates = _dates(80)
    daily = build_rebalance_mask(dates, "daily")
    assert daily is None or bool(np.all(daily))
    monthly = build_rebalance_mask(dates, "monthly")
    assert monthly is not None and int(monthly.sum()) < len(dates)
    weekly = build_rebalance_mask(dates, "weekly")
    assert weekly is not None
    assert int(weekly.sum()) > int(monthly.sum())
    with pytest.raises(ValueError, match="cadence"):
        build_rebalance_mask(dates, "hourly")


def test_off_rebalance_turnover_is_zero_and_cost_bounded():
    rng = np.random.default_rng(0)
    t, k = 120, 5
    rets = rng.normal(0.0004, 0.012, size=(t, k))
    fac = rng.normal(0.0, 0.008, size=(t, 4))
    dates = _dates(t)
    mask = month_end_mask(dates)
    arm = ArmSpec(id="eq", option_slots=0, equity_slots=k, delta_mode="off")
    friction = FrictionSpec(equity_bps=5.0, impact_c_eq=0.5)
    resid = freeze_residualizer(
        fit_ff4_residualizer(np.nanmean(rets, axis=1), fac, fold_id="cad"), "cad"
    )
    out = score_equal_weight(
        rets,
        factors=fac,
        arm=arm,
        friction=friction,
        residualizer=resid,
        rebalance_mask=mask,
        cadence="monthly",
    )
    # Off-rebalance days must have zero turnover. Series are indexed by
    # position, not absolute day; align via t_index (score_strategy no
    # longer pre-allocates a T-length array with an unwritten leading zero).
    t0 = int(out["t_index"][0])
    for i in range(1, t):
        if not mask[i]:
            pos = i - t0
            if 0 <= pos < out["turnover"].size:
                assert float(out["turnover"][pos]) == pytest.approx(0.0, **FLOAT_TOL)
    ann_to = annualized_turnover(out["turnover"])
    # EW monthly: entry once then hold → annualized turnover small.
    assert ann_to < 5.0
    # Annualized cost as fraction of NAV (sum of costs * 252 / T) under 3%.
    ann_cost = float(np.nansum(out["cost"]) * 252.0 / max(1, t))
    assert ann_cost < 0.03, f"annualized cost {ann_cost:.4f} >= 0.03"
