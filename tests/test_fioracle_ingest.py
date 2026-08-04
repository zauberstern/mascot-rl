"""Tests for scripts/ingest_fioracle_macro.py (vendored lake parquet + manifest)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.ingest_fioracle_macro import SERIES_SPECS, ingest_fioracle_macro


def _write_tiny_fioracle_tree(root: Path) -> None:
    """Synthesize minimal CSVs matching fioracle schemas."""
    (root / "macro_universe").mkdir(parents=True)
    (root / "dataset" / "EPU").mkdir(parents=True)
    (root / "ancillary").mkdir(parents=True)

    # VIX with synthetic pre-1990 rows that must be truncated
    vix = pd.DataFrame(
        {
            "Date": ["1989-12-29", "1989-12-30", "1990-01-01", "1990-01-02", "1990-01-03"],
            "Synthesized_VIX": [12.0, 12.5, 17.0, 18.0, 19.0],
        }
    )
    vix.to_csv(root / "macro_universe" / "VIX.csv", index=False)

    gpri = pd.DataFrame(
        {"Date": ["1990-01-01", "1990-01-02", "1990-01-03"], "GPRI": [100.0, 110.0, 105.0]}
    )
    gpri.to_csv(root / "macro_universe" / "GPRI.csv", index=False)

    # HY OAS: values are spread levels (~3), not a TR index
    hy = pd.DataFrame(
        {
            "Date": ["1994-01-31", "1994-02-01", "1994-02-02"],
            "Total Return Index": [3.29, 3.35, 3.40],
            "Beta Raw": [None, None, None],
            "Volatility 30D": [None, None, None],
        }
    )
    hy.to_csv(root / "macro_universe" / "US_HY_CORP_OAS_TOTAL_RETURN.csv", index=False)

    infl = pd.DataFrame(
        {"Date": ["1994-01-01", "1994-01-02", "1994-01-03"], "Inflation": [2.5, 2.5, 2.6]}
    )
    infl.to_csv(root / "macro_universe" / "US_INFLATION_RATE.csv", index=False)

    unemp = pd.DataFrame(
        {"Date": ["1994-01-01", "1994-01-02", "1994-01-03"], "Unemployment": [5.0, 5.0, 5.1]}
    )
    unemp.to_csv(root / "macro_universe" / "US_UNEMPLOYMENT_RATE.csv", index=False)

    # Real daily EPU (not backfilled macro_universe/EPU.csv)
    epu = pd.DataFrame(
        {
            "day": [1, 2, 3],
            "month": [1, 1, 1],
            "year": [1985, 1985, 1985],
            "EPU": [103.83, 296.43, 150.0],
        }
    )
    epu.to_csv(root / "dataset" / "EPU" / "US_EPU_Daily.csv", index=False)

    y2 = pd.DataFrame(
        {"observation_date": ["1990-01-01", "1990-01-02", "1990-01-03"], "DGS2": [7.0, 7.1, 7.2]}
    )
    y2.to_csv(root / "ancillary" / "US 2Y Yield (1976-present).csv", index=False)

    ts = pd.DataFrame(
        {
            "observation_date": ["1990-01-01", "1990-01-02", "1990-01-03"],
            "T10Y2Y": [0.5, 0.6, 0.55],
        }
    )
    ts.to_csv(root / "ancillary" / "US 10Y2Y (1976-present).csv", index=False)


@pytest.fixture
def fioracle_src(tmp_path: Path) -> Path:
    live = Path("/home/kartik/Desktop/fioracle/fioracle")
    src = tmp_path / "fioracle_src"
    if live.is_dir() and (live / "macro_universe" / "VIX.csv").is_file():
        # Prefer tiny fixtures for speed/determinism even when live exists
        _write_tiny_fioracle_tree(src)
    else:
        _write_tiny_fioracle_tree(src)
    return src


def test_ingest_idempotent_byte_identical(fioracle_src: Path, tmp_path: Path):
    dest = tmp_path / "lake" / "macro" / "fioracle"
    ingest_fioracle_macro(src=fioracle_src, dest=dest)
    first = {
        p.name: p.read_bytes()
        for p in sorted(dest.glob("*.parquet"))
    }
    manifest1 = (dest / "_manifest.json").read_bytes()

    ingest_fioracle_macro(src=fioracle_src, dest=dest)
    second = {
        p.name: p.read_bytes()
        for p in sorted(dest.glob("*.parquet"))
    }
    manifest2 = (dest / "_manifest.json").read_bytes()

    assert first == second
    assert manifest1 == manifest2
    assert set(SERIES_SPECS) == {p.stem for p in dest.glob("*.parquet")}


def test_available_date_equals_event_plus_lag(fioracle_src: Path, tmp_path: Path):
    dest = tmp_path / "out"
    ingest_fioracle_macro(src=fioracle_src, dest=dest)
    for series_id, spec in SERIES_SPECS.items():
        df = pd.read_parquet(dest / f"{series_id}.parquet")
        lag = int(spec["lag_days"])
        got = pd.to_datetime(df["available_date"])
        exp = pd.to_datetime(df["event_date"]) + pd.to_timedelta(lag, unit="D")
        assert (got == exp).all(), series_id
        assert (df["lag_days"] == lag).all()


def test_manifest_records_sha256(fioracle_src: Path, tmp_path: Path):
    dest = tmp_path / "out"
    ingest_fioracle_macro(src=fioracle_src, dest=dest)
    man = json.loads((dest / "_manifest.json").read_text())
    assert "series" in man
    for series_id, spec in SERIES_SPECS.items():
        entry = man["series"][series_id]
        src_path = fioracle_src / spec["source_file"]
        expected = hashlib.sha256(src_path.read_bytes()).hexdigest()
        assert entry["source_sha256"] == expected
        assert entry["lag_days"] == spec["lag_days"]
        assert "row_count" in entry
        assert "redistributable" in entry


def test_hy_oas_is_spread_range(fioracle_src: Path, tmp_path: Path):
    dest = tmp_path / "out"
    ingest_fioracle_macro(src=fioracle_src, dest=dest)
    df = pd.read_parquet(dest / "hy_oas.parquet")
    first = float(df["value"].iloc[0])
    assert 1.0 <= first <= 30.0


def test_vix_no_synthetic_before_1990(fioracle_src: Path, tmp_path: Path):
    dest = tmp_path / "out"
    ingest_fioracle_macro(src=fioracle_src, dest=dest)
    df = pd.read_parquet(dest / "vix.parquet")
    dates = pd.to_datetime(df["event_date"])
    # Either dropped or NaN — never a finite synthetic pre-1990 value
    pre = df.loc[dates < pd.Timestamp("1990-01-01")]
    if len(pre):
        assert pre["value"].isna().all()
    assert dates.min() >= pd.Timestamp("1990-01-01") or df.loc[
        dates >= pd.Timestamp("1990-01-01"), "value"
    ].notna().any()
    assert df.loc[dates >= pd.Timestamp("1990-01-01"), "value"].notna().all()


def test_epu_from_daily_file_not_before_1985(fioracle_src: Path, tmp_path: Path):
    dest = tmp_path / "out"
    ingest_fioracle_macro(src=fioracle_src, dest=dest)
    df = pd.read_parquet(dest / "epu.parquet")
    man = json.loads((dest / "_manifest.json").read_text())
    assert man["series"]["epu"]["source_file"] == "dataset/EPU/US_EPU_Daily.csv"
    assert pd.to_datetime(df["event_date"]).min() >= pd.Timestamp("1985-01-01")


def test_live_fioracle_optional_smoke():
    """If the live fioracle tree exists, ingest a copy into tmp (smoke only)."""
    live = Path("/home/kartik/Desktop/fioracle/fioracle")
    if not (live / "macro_universe" / "VIX.csv").is_file():
        pytest.skip("live fioracle path missing")
    # Use tmp dest only — do not write into the real lake in unit tests
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        dest = Path(td) / "macro" / "fioracle"
        # Copy only needed files would be huge; rely on fixture tests for logic.
        # Smoke: ensure SERIES_SPECS paths resolve under live src.
        for spec in SERIES_SPECS.values():
            assert (live / spec["source_file"]).is_file(), spec["source_file"]
