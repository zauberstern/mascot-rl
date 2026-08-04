"""Spine Compustat projection from full funda (no live WRDS)."""
from __future__ import annotations

import pandas as pd

from src.data.wrds_enrich import FUNDA_FULL_COLUMNS, project_spine_compustat_funda


def test_project_spine_compustat_funda_inner_join_secid() -> None:
    funda = pd.DataFrame(
        {
            "gvkey": ["001075", "001075", "002000"],
            "datadate": pd.to_datetime(["2020-12-31", "2021-12-31", "2020-12-31"]),
            "fyear": [2020, 2021, 2020],
            "tic": ["A", "A", "B"],
            "cusip": ["1", "1", "2"],
            "conm": ["A Co", "A Co", "B Co"],
            "dvc": [1.0, 1.1, 0.0],
            "dv": [1.0, 1.1, 0.0],
            "prcc_f": [10.0, 11.0, 5.0],
            "csho": [100.0, 100.0, 50.0],
            "at": [1000.0, 1100.0, 200.0],
            "sale": [500.0, 550.0, 80.0],
            "ni": [50.0, 55.0, 8.0],
        }
    )
    # pad extra full columns unused by projection
    for c in FUNDA_FULL_COLUMNS:
        if c not in funda.columns:
            funda[c] = None
    ccm = pd.DataFrame({"gvkey": ["001075", "002000"], "permno": [101, 202]})
    link = pd.DataFrame({"permno": [101], "secid": [999]})
    spine = project_spine_compustat_funda(funda, ccm, link)
    assert set(spine["secid"]) == {999}
    assert len(spine) == 2
    assert "dvc" in spine.columns and "at" in spine.columns


def test_project_spine_requires_thin_columns() -> None:
    funda = pd.DataFrame({"gvkey": ["1"]})
    try:
        project_spine_compustat_funda(funda, pd.DataFrame(), pd.DataFrame())
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "missing columns" in str(e)
