"""RED: periods_per_year must come from the realized date grid, not a constant 252."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_periods_per_year_from_dates_matches_grid_density() -> None:
    from src.eval.calendar_scaling import periods_per_year_from_dates

    # ~181 observations/year (punctured panel): 894 days over ~4.93 calendar years
    start = pd.Timestamp("2017-02-06")
    end = pd.Timestamp("2022-01-03")
    # Synthetic evenly spaced index with the same endpoints and count.
    dates = pd.date_range(start, end, periods=894)
    ppy = periods_per_year_from_dates(dates)
    expected = (894 - 1) / ((end - start).days / 365.25)
    assert ppy == pytest.approx(expected, rel=1e-6)
    assert ppy < 200.0  # clearly not a 252 trading-day year


def test_periods_per_year_rejects_short_grids() -> None:
    from src.eval.calendar_scaling import periods_per_year_from_dates

    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    with pytest.raises(ValueError, match="at least 30"):
        periods_per_year_from_dates(dates)


def test_eval_panel_meta_reports_coverage_below_intended() -> None:
    from src.eval.calendar_scaling import eval_panel_meta

    dates = pd.date_range("2017-02-06", "2022-01-03", periods=894)
    meta = eval_panel_meta(
        dates,
        k=100,
        intended_start="2014-01-01",
        intended_end="2024-12-31",
        rebalance_days=44,
    )
    assert meta["t"] == 894
    assert meta["k"] == 100
    assert meta["date_start"] == "2017-02-06"
    assert meta["date_end"] == "2022-01-03"
    assert meta["coverage_frac"] < 0.90
    assert meta["periods_per_year"] < 200.0
    assert meta["rebalance_days"] == 44
    assert meta["intended_start"] == "2014-01-01"
    assert meta["intended_end"] == "2024-12-31"
