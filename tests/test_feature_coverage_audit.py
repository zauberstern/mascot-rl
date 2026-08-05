"""Coverage audit and LSEG gap-ingest guards."""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pandas as pd

from mascotrl.data.feature_coverage import (
    CONSUMER_MAP,
    INTENTIONALLY_UNUSED,
    assert_coverage_ok,
    run_coverage_audit,
)
from mascotrl.data.lseg_overlay import copy_parallel_lseg, ingest_lseg_overlays


def test_consumer_map_and_unused_disjoint() -> None:
    overlap = set(CONSUMER_MAP) & set(INTENTIONALLY_UNUSED)
    assert not overlap


def test_coverage_audit_on_tmp_lake(tmp_path: Path) -> None:
    macro = tmp_path / "macro"
    macro.mkdir()
    # Minimal tables so audit can run offline.
    pd.DataFrame(
        {
            "secid": [1],
            "date": pd.to_datetime(["2020-01-02"]),
            "close": [10.0],
            "volume": [100.0],
            "return": [0.01],
            "cfadj": [1.0],
            "shrout": [1.0],
            "ticker": ["AAA"],
            "open": [9.9],
            "high": [10.1],
            "low": [9.8],
            "cusip": ["x"],
            "sic": [1],
            "index_flag": [0],
            "exchange_d": [1],
            "class": [""],
            "issue_type": [""],
            "industry_group": [0],
            "cfret": [0.01],
            "lseg_bid": [9.9],
            "lseg_ask": [10.1],
            "lseg_trdprc": [10.0],
            "lseg_open": [9.9],
            "lseg_high": [10.1],
            "lseg_low": [9.8],
            "lseg_trnvr": [1.0],
            "lseg_num_moves": [1],
            "lseg_acvol": [50.0],
            "lseg_vwap": [10.0],
            "lseg_vwap_vol": [40.0],
            "lseg_blkcount": [0],
            "lseg_blkvolum": [0.0],
            "lseg_trd_status": ["ok"],
            "lseg_ric": ["A"],
            "lseg_asof_ts": ["t"],
            "lseg_quoted_spread": [0.01],
        }
    ).to_parquet(macro / "sp500_sec.parquet", index=False)
    with mock.patch("mascotrl.data.feature_coverage.assert_lake_mounted", return_value=tmp_path):
        with mock.patch("mascotrl.data.feature_coverage.FLAT_TABLES", ("macro/sp500_sec.parquet",)):
            with mock.patch("mascotrl.data.feature_coverage.HIVE_TABLES", ()):
                report = run_coverage_audit(tmp_path)
    assert report["ok"] is True
    assert report["n_missing"] == 0


def test_assert_coverage_ok_live_lake() -> None:
    """Live lake must classify every column (mount required)."""
    from mascotrl.data.paths import LAKE_ROOT

    if not LAKE_ROOT.exists():
        return
    report = assert_coverage_ok()
    assert report["ok"] is True
    assert report["n_columns"] > 100


def test_ingest_copies_corax_and_index_vol_rates(tmp_path: Path) -> None:
    raw = tmp_path / "lseg"
    macro_raw = raw / "macro"
    macro_raw.mkdir(parents=True)
    lake_macro = tmp_path / "lake_macro"
    lake_macro.mkdir()
    # Minimal overlays targets.
    pd.DataFrame(
        {
            "secid": [1],
            "date": pd.to_datetime(["2020-01-02"]),
            "close": [10.0],
            "return": [0.01],
            "volume": [1.0],
            "cfadj": [1.0],
        }
    ).to_parquet(lake_macro / "sp500_sec.parquet", index=False)
    pd.DataFrame({"date": pd.to_datetime(["2020-01-02"]), "dtb3": [1.0]}).to_parquet(
        lake_macro / "interest_rate.parquet", index=False
    )
    ohlc_cols = {
        "secid": [1],
        "date": pd.to_datetime(["2020-01-02"]),
        "BID": [9.9],
        "ASK": [10.1],
        "TRDPRC_1": [10.0],
        "OPEN_PRC": [9.8],
        "HIGH_1": [10.2],
        "LOW_1": [9.7],
        "TRNOVR_UNS": [1.0],
        "NUM_MOVES": [1],
        "ACVOL_UNS": [50.0],
        "VWAP": [10.0],
        "VWAP_VOL": [40.0],
        "BLKCOUNT": [0],
        "BLKVOLUM": [0.0],
        "TRD_STATUS": ["ok"],
        "ric": ["A.O"],
        "asof_ts": ["t"],
    }
    pd.DataFrame(ohlc_cols).to_parquet(macro_raw / "lseg_eq_ohlc_corax.parquet", index=False)
    pd.DataFrame(ohlc_cols).to_parquet(macro_raw / "lseg_eq_ohlc_unadj.parquet", index=False)
    pd.DataFrame(
        {
            "secid": [1],
            "date": pd.to_datetime(["2020-01-02"]),
            "quoted_spread": [0.01],
            "ric": ["A.O"],
            "asof_ts": ["t"],
        }
    ).to_parquet(macro_raw / "lseg_eq_spread.parquet", index=False)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"]),
            "ric": ["US2YT=RR"],
            "YLDTOMAT": [1.0],
            "FIXING_1": [1.0],
            "asof_ts": ["t"],
        }
    ).to_parquet(macro_raw / "lseg_index_vol_rates.parquet", index=False)
    for name in (
        "lseg_eq_size.parquet",
        "lseg_spx_pit.parquet",
        "lseg_gics.parquet",
    ):
        if name == "lseg_gics.parquet":
            pd.DataFrame(
                {
                    "Instrument": ["A.O"],
                    "TR.GICSIndustry": ["x"],
                    "TR.GICSSector": ["y"],
                    "TR.GICSSubIndustry": ["z"],
                    "asof_ts": ["t"],
                }
            ).to_parquet(macro_raw / name, index=False)
        elif name == "lseg_spx_pit.parquet":
            pd.DataFrame(
                {
                    "Instrument": ["SPX"],
                    "Constituent RIC": ["A.O"],
                    "pit_date": pd.to_datetime(["2020-01-02"]),
                    "asof_ts": ["t"],
                }
            ).to_parquet(macro_raw / name, index=False)
        else:
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2020-01-02"]),
                    "Company Market Cap": [1.0],
                    "Free Float": [1.0],
                    "Free Float (Percent)": [1.0],
                    "Outstanding Shares": [1.0],
                    "secid": [1],
                    "ric": ["A.O"],
                    "asof_ts": ["t"],
                }
            ).to_parquet(macro_raw / name, index=False)

    ingest_lseg_overlays(lseg_raw=raw, lake_macro=lake_macro)
    assert (lake_macro / "lseg_eq_ohlc_corax.parquet").is_file()
    assert (lake_macro / "lseg_index_vol_rates.parquet").is_file()


def test_copy_parallel_lseg_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "mascotrl.parquet"
    dest = tmp_path / "dest.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(src, index=False)
    copy_parallel_lseg(src=src, dest=dest)
    assert dest.is_file()
