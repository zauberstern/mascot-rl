"""Availability-mask estimand: keep all eval dates; mask inactive names."""
from __future__ import annotations

import numpy as np
import pandas as pd


def test_wide_returns_availability_keeps_full_calendar_span() -> None:
    from src.eval.equity_substrate import _wide_returns_with_availability

    dates = pd.bdate_range("2014-01-02", "2014-03-31")
    rows = []
    # Name A present whole window; name B missing first month.
    for d in dates:
        rows.append({"date": d, "secid": 1, "return": 0.001})
        if d >= pd.Timestamp("2014-02-01"):
            rows.append({"date": d, "secid": 2, "return": 0.001})
    df = pd.DataFrame(rows)
    rets, secids, idx, avail = _wide_returns_with_availability(
        df, start="2014-01-02", end="2014-03-31", min_cov=0.1, ffill_limit=0
    )
    assert list(secids) == [1, 2] or set(secids) == {1, 2}
    assert len(idx) == len(dates)
    assert avail.shape == rets.shape
    # Early dates: name B unavailable
    early = idx < pd.Timestamp("2014-02-01")
    assert avail[early, list(secids).index(2)].sum() == 0
    assert avail[~early, list(secids).index(2)].all()
