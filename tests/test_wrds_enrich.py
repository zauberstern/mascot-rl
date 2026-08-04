"""TDD: WRDS/local enrichment materializes ADV + CRSP-OM link."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.wrds_enrich import (
    build_adv_panel_from_crsp_and_link,
    load_adv_for_secid,
    write_enrichment_parquets,
)


def test_build_adv_from_crsp_and_link(tmp_path: Path) -> None:
    crsp = pd.DataFrame(
        {
            "PERMNO": [101, 101, 202, 202],
            "date": pd.to_datetime(
                ["2020-01-02", "2020-01-03", "2020-01-02", "2020-01-03"]
            ),
            "PRC": [-10.0, 11.0, 20.0, 21.0],  # CRSP negative PRC = bid/ask avg sign
            "VOL": [1_000_000, 2_000_000, 500_000, 600_000],
            "NCUSIP": ["A"] * 4,
        }
    )
    link = pd.DataFrame(
        {
            "secid": [1001.0, 1002.0],
            "permno": [101, 202],
            "sdate": pd.to_datetime(["1996-01-01", "1996-01-01"]),
            "edate": pd.to_datetime(["2025-12-31", "2025-12-31"]),
            "score": [1.0, 1.0],
        }
    )
    adv = build_adv_panel_from_crsp_and_link(crsp, link)
    assert set(adv.columns) >= {"date", "secid", "permno", "adv"}
    assert len(adv) == 4
    row = adv[(adv["secid"] == 1001) & (adv["date"] == pd.Timestamp("2020-01-02"))].iloc[
        0
    ]
    assert row["adv"] == pytest.approx(10.0 * 1_000_000)


def test_load_adv_for_secid_median(tmp_path: Path) -> None:
    adv = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-04"]),
            "secid": [42, 42, 42],
            "permno": [1, 1, 1],
            "adv": [1e8, 2e8, 3e8],
        }
    )
    path = tmp_path / "crsp_om_adv.parquet"
    adv.to_parquet(path, index=False)
    v = load_adv_for_secid(path, secid=42, start="2020-01-02", end="2020-01-04")
    assert v == pytest.approx(2e8)


def test_write_enrichment_parquets(tmp_path: Path) -> None:
    link = pd.DataFrame(
        {
            "secid": [1.0],
            "permno": [10],
            "sdate": pd.to_datetime(["2000-01-01"]),
            "edate": pd.to_datetime(["2030-01-01"]),
            "score": [1.0],
        }
    )
    adv = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"]),
            "secid": [1],
            "permno": [10],
            "adv": [1e9],
            "prc": [10.0],
            "vol": [1e8],
        }
    )
    comp = pd.DataFrame(
        {
            "gvkey": ["001"],
            "datadate": pd.to_datetime(["2020-12-31"]),
            "tic": ["X"],
            "dvc": [1.0],
        }
    )
    paths = write_enrichment_parquets(
        tmp_path, link=link, adv=adv, compustat=comp
    )
    assert paths["link"].exists()
    assert paths["adv"].exists()
    assert paths["compustat"].exists()
    assert pd.read_parquet(paths["adv"])["adv"].iloc[0] == pytest.approx(1e9)
