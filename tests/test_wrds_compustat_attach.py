"""Compustat attach stamps episode residuals (no live WRDS)."""
from __future__ import annotations

import pytest
from tests.conftest import FLOAT_TOL

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.data.wrds_enrich import attach_episode_compustat


def test_attach_compustat_tags_dvc(tmp_path: Path) -> None:
    macro = tmp_path / "macro"
    macro.mkdir()
    pd.DataFrame(
        {
            "gvkey": ["001"],
            "datadate": [pd.Timestamp("2020-12-31")],
            "fyear": [2020],
            "tic": ["AAA"],
            "cusip": ["111"],
            "conm": ["AAA CORP"],
            "dvc": [12.5],
            "dv": [12.5],
            "prcc_f": [10.0],
            "csho": [100.0],
            "at": [1.0],
            "sale": [1.0],
            "ni": [1.0],
            "secid": [42],
        }
    ).to_parquet(macro / "compustat_funda_enrich.parquet", index=False)

    ep = SimpleNamespace(
        secid="42",
        dates=["2021-01-04", "2021-01-05"],
        estimand_residuals={},
    )
    stats = attach_episode_compustat([ep], lake_base_dir=tmp_path)
    assert stats["n_with_compustat"] == pytest.approx(1.0, **FLOAT_TOL)
    assert ep.estimand_residuals["compustat_state"] == "compustat_funda_enrich"
    assert ep.estimand_residuals["american_residual"] == "compustat_dvc_positive"
    assert ep.estimand_residuals["dvc_annual"] == pytest.approx(12.5, **FLOAT_TOL)
