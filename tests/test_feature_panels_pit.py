"""PIT lag tests for feature panel materializers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.feature_panels import (
    COMPUSTAT_LAG_DAYS,
    SHORT_INTEREST_LAG_DAYS,
    load_compustat_long,
    load_ibes_ratios_long,
    load_short_interest_long,
)


def test_short_interest_availability_lag(tmp_path: Path) -> None:
    macro = tmp_path / "macro" / "p3"
    macro.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"]),
            "Short Interest": [1e6],
            "Short Interest Pct": [2.5],
            "secid": [1],
            "ric": ["A"],
        }
    ).to_parquet(macro / "lseg_short_interest.parquet", index=False)
    out = load_short_interest_long(tmp_path, "2020-01-01", "2020-02-01")
    assert len(out) == 1
    assert out.iloc[0]["date"] == pd.Timestamp("2020-01-02") + pd.Timedelta(
        days=SHORT_INTEREST_LAG_DAYS
    )


def test_compustat_120d_lag(tmp_path: Path) -> None:
    macro = tmp_path / "macro"
    macro.mkdir()
    pd.DataFrame(
        {
            "gvkey": [1],
            "datadate": pd.to_datetime(["2019-12-31"]),
            "fyear": [2019],
            "tic": ["A"],
            "cusip": ["x"],
            "conm": ["A"],
            "dvc": [1.0],
            "dv": [1.0],
            "prcc_f": [10.0],
            "csho": [1.0],
            "at": [100.0],
            "sale": [50.0],
            "ni": [5.0],
            "permno": [1],
            "secid": [1],
        }
    ).to_parquet(macro / "compustat_funda_enrich.parquet", index=False)
    # Daily sec calendar spanning lag boundary.
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    pd.DataFrame(
        {
            "secid": [1] * len(dates),
            "date": dates,
            "close": [10.0] * len(dates),
            "volume": [1.0] * len(dates),
            "cfadj": [1.0] * len(dates),
            "open": [10.0] * len(dates),
            "high": [10.1] * len(dates),
            "low": [9.9] * len(dates),
        }
    ).to_parquet(macro / "sp500_sec.parquet", index=False)
    out = load_compustat_long(tmp_path, "2020-01-01", "2020-12-31")
    avail = pd.Timestamp("2019-12-31") + pd.Timedelta(days=COMPUSTAT_LAG_DAYS)
    before = out[out["date"] < avail]
    after = out[out["date"] >= avail]
    assert before["ni_at"].isna().all()
    assert after["ni_at"].notna().any()


def test_ibes_public_date_pit(tmp_path: Path) -> None:
    macro = tmp_path / "macro"
    macro.mkdir()
    pd.DataFrame(
        {
            "gvkey": [1, 1],
            "permno": [10, 10],
            "adate": pd.to_datetime(["2019-01-01", "2019-01-01"]),
            "qdate": pd.to_datetime(["2019-01-01", "2019-01-01"]),
            "public_date": pd.to_datetime(["2020-06-01", "2020-01-15"]),
            "bm": [0.5, 0.4],
            "pe_exi": [10.0, 12.0],
            "ps": [1.0, 1.1],
            "pcf": [1.0, 1.1],
            "dpr": [0.1, 0.1],
            "npm": [0.1, 0.1],
            "gpm": [0.2, 0.2],
            "roa": [0.05, 0.05],
            "roe": [0.1, 0.1],
            "cfm": [0.1, 0.1],
            "evm": [1.0, 1.0],
            "CAPEI": [15.0, 16.0],
        }
    ).to_parquet(macro / "ibes_financial_ratios.parquet", index=False)
    pd.DataFrame(
        {
            "secid": [1],
            "permno": [10],
            "sdate": pd.to_datetime(["2000-01-01"]),
            "edate": pd.to_datetime(["2030-01-01"]),
            "score": [1],
        }
    ).to_parquet(macro / "crsp_optionm_link.parquet", index=False)
    dates = pd.to_datetime(["2020-01-10", "2020-02-01", "2020-07-01"])
    pd.DataFrame(
        {
            "secid": [1, 1, 1],
            "date": dates,
            "close": [10.0, 10.0, 10.0],
            "volume": [1.0, 1.0, 1.0],
            "cfadj": [1.0, 1.0, 1.0],
            "open": [10.0, 10.0, 10.0],
            "high": [10.0, 10.0, 10.0],
            "low": [10.0, 10.0, 10.0],
        }
    ).to_parquet(macro / "sp500_sec.parquet", index=False)
    out = load_ibes_ratios_long(tmp_path, "2020-01-01", "2020-12-31")
    by_date = out.set_index("date")["bm"]
    # Before any public_date: NaN
    assert np.isnan(by_date.loc[pd.Timestamp("2020-01-10")])
    # After Jan 15 public: bm=0.4
    assert by_date.loc[pd.Timestamp("2020-02-01")] == pytest.approx(0.4)
    # After June public: bm=0.5 (later filing)
    assert by_date.loc[pd.Timestamp("2020-07-01")] == pytest.approx(0.5)
