"""Orientation benches: lake alignment + never enter BASELINE_NAMES."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mascotrl.eval.baselines import BASELINE_NAMES
from mascotrl.eval.orientation_benchmarks import (
    ORIENTATION_NAMES,
    align_to_dates,
    run_and_attach_orientation_benchmarks,
    run_orientation_benchmarks,
)


def test_orientation_names_not_in_baseline_names():
    for name in ORIENTATION_NAMES:
        assert name not in BASELINE_NAMES


def test_align_and_pack_synthetic(tmp_path: Path):
    dates = pd.bdate_range("2022-01-03", periods=20)
    # Build tiny fake lake
    lake = tmp_path / "lake"
    (lake / "macro").mkdir(parents=True)
    eq = pd.DataFrame(
        {
            "date": np.repeat(dates, 3),
            "vwretd": np.tile(np.linspace(-0.01, 0.01, len(dates)), 3),
            "sprtrn": np.tile(np.linspace(-0.01, 0.01, len(dates)), 3),
        }
    )
    eq.to_parquet(lake / "macro" / "sp500_prices.parquet")
    ir = pd.DataFrame(
        {
            "date": dates,
            "effr": [5.0] * len(dates),
            "dtb3": [4.5] * len(dates),
        }
    )
    ir.to_parquet(lake / "macro" / "interest_rate.parquet")

    suite = run_orientation_benchmarks(dates.tolist(), lake_base_dir=lake)
    assert suite["status"] == "ok"
    assert suite["coverage"]["equity_market"] == pytest.approx(1.0)
    assert suite["coverage"]["cash_rf"] == pytest.approx(1.0)
    assert np.isfinite(suite["summary"]["equity_market"]["sharpe"])
    # Cash daily = 5%/252 — nearly constant → Sharpe undefined/huge; still finite n_days.
    assert suite["summary"]["cash_rf"]["n_days"] == len(dates)
    assert not np.isfinite(suite["summary"]["cash_rf"]["sharpe"])  # near-deterministic
    assert suite["summary"]["cash_rf"]["mean_ann"] == pytest.approx(5.0 / 100.0)

    report = {
        "historical_oos": {
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "summary": {"happo": {"sharpe": 1.5}},
            "friction_applied": False,
        }
    }
    run_and_attach_orientation_benchmarks(report, lake_base_dir=lake)
    assert "orientation_benchmarks" in report
    assert report["orientation_lead"]["happo_sharpe"] == pytest.approx(1.5)


def test_align_to_dates_partial_coverage():
    s = pd.Series(
        [0.01, 0.02],
        index=pd.to_datetime(["2022-01-03", "2022-01-04"]),
    )
    vals, _, cov = align_to_dates(
        s, ["2022-01-03", "2022-01-04", "2022-01-05"]
    )
    assert cov == pytest.approx(2 / 3)
    assert np.isnan(vals[2])
