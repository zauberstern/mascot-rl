"""Unit tests for OptionMetrics enrichment helpers (no live WRDS)."""
from __future__ import annotations

import pytest
from tests.conftest import FLOAT_TOL

from pathlib import Path

import pandas as pd

from mascotrl.data.om_enrich import build_option_adv_from_opvold, median_borrow_bps


def test_build_option_adv_prefers_aggregate_rows() -> None:
    opv = pd.DataFrame(
        {
            "secid": [1, 1, 1, 1],
            "date": pd.to_datetime(
                ["2020-01-02", "2020-01-02", "2020-01-02", "2020-01-03"]
            ),
            "cp_flag": ["C", "P", None, None],
            "volume": [10.0, 20.0, 100.0, 50.0],
            "open_interest": [1.0, 2.0, 9.0, 8.0],
        }
    )
    adv = build_option_adv_from_opvold(opv)
    assert len(adv) == 2
    row = adv[adv["date"] == pd.Timestamp("2020-01-02")].iloc[0]
    assert row["opt_volume"] == pytest.approx(100.0, **FLOAT_TOL)
    assert row["opt_open_interest"] == pytest.approx(9.0, **FLOAT_TOL)


def test_median_borrow_bps(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "secid": [7, 7, 7],
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-04"]),
            "borrowrate": [0.5, 1.0, 1.5],  # percent
            "days": [30, 30, 30],
        }
    )
    path = tmp_path / "om_stdbrte.parquet"
    df.to_parquet(path, index=False)
    bps = median_borrow_bps(path, secid=7)
    assert bps == pytest.approx(100.0, **FLOAT_TOL)# median 1.0% → 100 bps
