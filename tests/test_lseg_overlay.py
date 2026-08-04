"""LSEG overlay must not overwrite OM/CRSP SoT columns."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.lseg_overlay import (
    P3_REFUSED,
    overlay_interest_rate,
    overlay_sp500_sec,
    refuse_p3_path,
)


def test_overlay_sp500_sec_adds_lseg_cols_without_touching_om(tmp_path: Path) -> None:
    lake = tmp_path / "macro"
    lake.mkdir()
    sec = pd.DataFrame(
        {
            "secid": [1, 1],
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "close": [10.0, 11.0],
            "return": [0.01, 0.02],
            "volume": [100.0, 110.0],
            "cfadj": [1.0, 1.0],
        }
    )
    sec.to_parquet(lake / "sp500_sec.parquet", index=False)
    ohlc = pd.DataFrame(
        {
            "secid": [1, 1],
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "BID": [9.9, 10.9],
            "ASK": [10.1, 11.1],
            "TRDPRC_1": [10.0, 11.0],
            "OPEN_PRC": [9.8, 10.8],
            "HIGH_1": [10.2, 11.2],
            "LOW_1": [9.7, 10.7],
            "TRNOVR_UNS": [1.0, 2.0],
            "NUM_MOVES": [3, 4],
            "ACVOL_UNS": [50.0, 60.0],
            "VWAP": [10.0, 11.0],
            "VWAP_VOL": [40.0, 50.0],
            "BLKCOUNT": [0, 1],
            "BLKVOLUM": [0.0, 5.0],
            "TRD_STATUS": ["ok", "ok"],
            "ric": ["A.O", "A.O"],
            "asof_ts": ["2026-08-18", "2026-08-18"],
        }
    )
    spread = pd.DataFrame(
        {
            "secid": [1, 1],
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "quoted_spread": [0.01, 0.009],
            "ric": ["A.O", "A.O"],
            "asof_ts": ["2026-08-18", "2026-08-18"],
        }
    )
    ohlc.to_parquet(tmp_path / "ohlc.parquet", index=False)
    spread.to_parquet(tmp_path / "spread.parquet", index=False)
    out = overlay_sp500_sec(
        lake / "sp500_sec.parquet",
        ohlc_path=tmp_path / "ohlc.parquet",
        spread_path=tmp_path / "spread.parquet",
    )
    assert out["close"].tolist() == [10.0, 11.0]
    assert out["return"].tolist() == [0.01, 0.02]
    assert "lseg_bid" in out.columns
    assert "lseg_quoted_spread" in out.columns
    assert float(out.loc[0, "lseg_bid"]) == pytest.approx(9.9)


def test_overlay_interest_rate_keeps_dtb3_drops_oas(tmp_path: Path) -> None:
    rates = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "dtb3": [1.5, 1.6],
            "sofr": [1.4, 1.5],
            "effr": [1.55, 1.65],
        }
    )
    rates.to_parquet(tmp_path / "interest_rate.parquet", index=False)
    lseg = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-02"]),
            "ric": ["US2YT=RR", "US10YT=RR", "USDSOFR="],
            "YLDTOMAT": [1.1, 1.8, None],
            "FIXING_1": [None, None, 1.52],
            "OAS_BID": [9.0, 9.0, 9.0],
            "ZSPREAD": [8.0, 8.0, 8.0],
            "INT_CDS": [7.0, 7.0, 7.0],
            "asof_ts": ["2026-08-18"] * 3,
        }
    )
    lseg.to_parquet(tmp_path / "lseg_rates.parquet", index=False)
    out = overlay_interest_rate(
        tmp_path / "interest_rate.parquet",
        lseg_path=tmp_path / "lseg_rates.parquet",
    )
    assert out["dtb3"].tolist() == [1.5, 1.6]
    assert "lseg_us2y" in out.columns
    assert "OAS_BID" not in out.columns
    assert float(out.loc[out["date"] == pd.Timestamp("2020-01-02"), "lseg_us2y"].iloc[0]) == pytest.approx(1.1)


def test_refuse_p3_paths() -> None:
    for name in P3_REFUSED:
        with pytest.raises(ValueError, match="P3"):
            refuse_p3_path(Path("macro/lseg_p3") / name)
