"""PIT contract and causal derived features for fioracle macro panel."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mascotrl.data.fioracle_macro import (
    FIORACLE_FEATURE_COLUMNS,
    build_fioracle_feature_frame,
    load_fioracle_macro,
)


def _write_series(
    dest: Path,
    series_id: str,
    *,
    event_dates: list[str],
    values: list[float],
    lag_days: int,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    ev = pd.to_datetime(event_dates)
    avail = ev + pd.to_timedelta(lag_days, unit="D")
    df = pd.DataFrame(
        {
            "event_date": ev.date,
            "available_date": avail.date,
            "value": values,
            "series_id": series_id,
            "source_file": f"fixture/{series_id}.csv",
            "source_sha256": "abc",
            "lag_days": np.int32(lag_days),
        }
    )
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, dest / f"{series_id}.parquet", compression="zstd")


@pytest.fixture
def fioracle_lake(tmp_path: Path) -> Path:
    lake = tmp_path / "macro" / "fioracle"
    # Monthly-ish inflation with 15-day lag: event 2020-01-01 available 2020-01-16
    _write_series(
        lake,
        "inflation",
        event_dates=["2020-01-01", "2020-02-01"],
        values=[2.0, 2.2],
        lag_days=15,
    )
    _write_series(
        lake,
        "vix",
        event_dates=["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-16", "2020-01-17"],
        values=[12.0, 13.0, 14.0, 15.0, 16.0],
        lag_days=1,
    )
    _write_series(
        lake,
        "hy_oas",
        event_dates=["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-16", "2020-01-17"],
        values=[3.0, 3.1, 3.2, 3.3, 3.4],
        lag_days=1,
    )
    _write_series(
        lake,
        "term_spread",
        event_dates=["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-16", "2020-01-17"],
        values=[0.5, 0.5, 0.5, 0.4, 0.4],
        lag_days=0,
    )
    _write_series(
        lake,
        "epu",
        event_dates=["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-16", "2020-01-17"],
        values=[100.0, 110.0, 120.0, 130.0, 140.0],
        lag_days=7,
    )
    _write_series(
        lake,
        "gpri",
        event_dates=["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-16", "2020-01-17"],
        values=[80.0, 81.0, 82.0, 83.0, 84.0],
        lag_days=5,
    )
    _write_series(
        lake,
        "unemployment",
        event_dates=["2020-01-01", "2020-02-01"],
        values=[4.0, 4.1],
        lag_days=7,
    )
    _write_series(
        lake,
        "yield_2y",
        event_dates=["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-16", "2020-01-17"],
        values=[1.5, 1.5, 1.6, 1.6, 1.7],
        lag_days=0,
    )
    return lake


def test_monthly_lag15_invisible_then_visible(fioracle_lake: Path):
    # event 2020-01-01, lag 15 -> available 2020-01-16
    before = load_fioracle_macro(
        lake_root=fioracle_lake.parent.parent,
        start_date="2020-01-15",
        end_date="2020-01-15",
        series=["inflation"],
        use_available_date=True,
        lake_subdir="macro/fioracle",
    )
    assert "inflation" in before.columns
    assert pd.isna(before.loc[pd.Timestamp("2020-01-15"), "inflation"])

    on = load_fioracle_macro(
        lake_root=fioracle_lake.parent.parent,
        start_date="2020-01-16",
        end_date="2020-01-16",
        series=["inflation"],
        use_available_date=True,
        lake_subdir="macro/fioracle",
    )
    assert float(on.loc[pd.Timestamp("2020-01-16"), "inflation"]) == pytest.approx(2.0)


def test_use_available_date_false_differs(fioracle_lake: Path):
    pit = load_fioracle_macro(
        lake_root=fioracle_lake.parent.parent,
        start_date="2020-01-01",
        end_date="2020-01-20",
        series=["inflation"],
        use_available_date=True,
        lake_subdir="macro/fioracle",
    )
    leak = load_fioracle_macro(
        lake_root=fioracle_lake.parent.parent,
        start_date="2020-01-01",
        end_date="2020-01-20",
        series=["inflation"],
        use_available_date=False,
        lake_subdir="macro/fioracle",
    )
    assert not pit["inflation"].equals(leak["inflation"])


def test_rolling_z_causal_when_future_deleted(fioracle_lake: Path):
    # Need longer vix history for rolling z
    lake = fioracle_lake
    dates = pd.bdate_range("2019-01-01", periods=300)
    rng = np.random.default_rng(0)
    vals = 15.0 + rng.normal(0, 1, size=len(dates)).cumsum() * 0.01
    _write_series(
        lake,
        "vix",
        event_dates=[d.strftime("%Y-%m-%d") for d in dates],
        values=vals.tolist(),
        lag_days=0,
    )
    # Fill other required series with constants so feature frame builds
    for sid, lag in [
        ("hy_oas", 0),
        ("term_spread", 0),
        ("epu", 0),
        ("gpri", 0),
        ("unemployment", 0),
        ("inflation", 0),
        ("yield_2y", 0),
    ]:
        _write_series(
            lake,
            sid,
            event_dates=[d.strftime("%Y-%m-%d") for d in dates],
            values=(np.full(len(dates), 3.0)).tolist(),
            lag_days=lag,
        )

    full = load_fioracle_macro(
        lake_root=lake.parent.parent,
        start_date="2019-01-01",
        end_date=dates[-1].strftime("%Y-%m-%d"),
        use_available_date=True,
        lake_subdir="macro/fioracle",
    )
    feats_full = build_fioracle_feature_frame(full)
    t_cut = dates[200]
    truncated = full.loc[:t_cut].copy()
    feats_trunc = build_fioracle_feature_frame(truncated)

    col = "vix_z_252"
    assert col in FIORACLE_FEATURE_COLUMNS
    a = feats_full.loc[:t_cut, col]
    b = feats_trunc.loc[:t_cut, col]
    # Equal where both finite
    mask = a.notna() & b.notna()
    assert mask.any()
    np.testing.assert_allclose(a[mask].to_numpy(), b[mask].to_numpy(), rtol=1e-10, atol=1e-10)
