"""short_interest_ratio builder: flag-default-off, strict PIT ffill, NaN policy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mascotrl.data.short_interest import build_short_interest_ratio


def test_short_interest_disabled_by_default_returns_empty():
    table = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "secid": [1, 1],
            "Short Interest Pct": [1.5, 2.0],
        }
    )
    out = build_short_interest_ratio(table, dates=table["date"], secids=[1])
    assert out.empty or out["short_interest_ratio"].isna().all()


def test_short_interest_ffill_pit_no_bfill_no_zeros(tmp_path):
    table = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-03-31"]),
            "secid": [101, 101],
            "Short Interest Pct": [1.25, 2.50],
        }
    )
    dates = pd.to_datetime(
        ["2020-01-15", "2020-01-31", "2020-02-15", "2020-03-31", "2020-04-15"]
    )
    out = build_short_interest_ratio(
        table,
        dates=dates,
        secids=[101],
        enabled=True,
    )
    assert list(out.columns) == ["secid", "date", "short_interest_ratio"]
    wide = out.pivot(index="date", columns="secid", values="short_interest_ratio")
    # Before first disclosure: NaN (not zero, not bfill from future).
    assert np.isnan(wide.loc[dates[0], 101])
    assert wide.loc[dates[1], 101] == pytest.approx(1.25)
    # Mid-gap ffill from past disclosure only.
    assert wide.loc[dates[2], 101] == pytest.approx(1.25)
    assert wide.loc[dates[3], 101] == pytest.approx(2.50)
    assert wide.loc[dates[4], 101] == pytest.approx(2.50)
    assert not (wide.to_numpy() == 0.0).any()


def test_short_interest_reads_p3_parquet_when_enabled(tmp_path):
    p = tmp_path / "lseg_short_interest.parquet"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2010-06-30"]),
            "secid": [55],
            "Short Interest Pct": [3.0],
            "ric": ["X"],
        }
    ).to_parquet(p)
    dates = pd.to_datetime(["2010-06-30", "2010-07-01"])
    out = build_short_interest_ratio(
        p,
        dates=dates,
        secids=[55],
        enabled=True,
    )
    assert out.loc[out["date"] == dates[0], "short_interest_ratio"].iloc[0] == pytest.approx(3.0)
