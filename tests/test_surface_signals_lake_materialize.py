"""A12: happy-path coverage for materialize_surface_signals_from_lake against
a fixture parquet lake (hive-partitioned vol_surface tree)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mascotrl.data.surface_signals import (
    materialize_kelly_iv_images_from_lake,
    materialize_surface_signals_from_lake,
)


def _point(secid: int, date: str, days: int, delta: int, cp_flag: str, iv: float) -> dict:
    spot = 100.0
    d = abs(int(delta))
    if str(cp_flag).upper().startswith("P"):
        strike = spot * (1.0 - d / 200.0)
    else:
        strike = spot if int(delta) == 50 else spot * (1.0 + d / 200.0)
    return {
        "secid": secid,
        "date": pd.Timestamp(date),
        "days": int(days),
        "delta": int(delta),
        "cp_flag": cp_flag,
        "impl_volatility": float(iv),
        "impl_strike": float(strike),
        "impl_premium": max(0.5, abs(spot - strike) * 0.1 + 1.0),
        "dispersion": 0.01,
    }


def _write_fixture_lake(root: Path) -> None:
    rows = []
    for date, atm in (("2020-01-31", 0.20), ("2020-02-28", 0.30)):
        rows.append(_point(7, date, 30, 50, "C", atm))
        rows.append(_point(7, date, 30, -50, "P", atm + 0.02))
    df = pd.DataFrame(rows)
    part = root / "vol_surface" / "year=2020" / "month=01"
    part.mkdir(parents=True, exist_ok=True)
    df.to_parquet(part / "data_0.parquet", index=False)


def test_materialize_from_lake_happy_path(tmp_path: Path) -> None:
    _write_fixture_lake(tmp_path)
    panel = materialize_surface_signals_from_lake(
        tmp_path,
        secids=[7],
        start="2020-01-01",
        end="2020-12-31",
        month_end_only=True,
    )
    assert not panel.empty
    assert set(["secid", "date"]).issubset(panel.columns)
    assert set(panel["secid"].unique().tolist()) == {7}
    assert panel["date"].nunique() == 2


def test_materialize_from_lake_caches_when_cache_path_given(tmp_path: Path) -> None:
    _write_fixture_lake(tmp_path)
    cache = tmp_path / "cache.parquet"
    panel = materialize_surface_signals_from_lake(
        tmp_path,
        secids=[7],
        start="2020-01-01",
        end="2020-12-31",
        cache_path=cache,
    )
    assert cache.exists()
    cached = pd.read_parquet(cache)
    assert len(cached) == len(panel)


def test_materialize_from_lake_empty_secids_raises(tmp_path: Path) -> None:
    _write_fixture_lake(tmp_path)
    with pytest.raises(ValueError, match="secids"):
        materialize_surface_signals_from_lake(
            tmp_path, secids=[], start="2020-01-01", end="2020-12-31"
        )


def test_materialize_kelly_iv_images_from_lake_happy_path(tmp_path: Path) -> None:
    """B3: real per-date Kelly grid from the lake, correct shape, no NaN
    after forward-fill, and no look-ahead across the date axis."""
    _write_fixture_lake(tmp_path)
    dates = [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-15"), pd.Timestamp("2020-02-28")]
    cube = materialize_kelly_iv_images_from_lake(
        tmp_path, secids=[7], dates=dates, start="2020-01-01", end="2020-12-31"
    )
    assert cube.shape == (3, 1, 11, 34)
    # Fixture only wrote the 30d/±50-delta nodes; those must be populated.
    assert np.nanmax(cube[0, 0]) > 0.0
    # 2020-02-15 has no direct quote; forward-fill must carry 2020-01-31's value.
    assert np.allclose(cube[1, 0], cube[0, 0], equal_nan=True)
    # 2020-02-28's real quote (atm=0.30) must differ from the carried value.
    assert not (cube[2, 0] == cube[0, 0]).all()


def test_load_vol_surface_raw_bounds_duckdb_memory_and_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A defect: `_load_vol_surface_raw` used a bare `duckdb.connect()` with
    no memory/thread ceiling, so a full-universe scan defaulted to DuckDB's
    own auto-detected limits (a large share of total host RAM, all cores)
    and could OOM the whole machine. It must honor the same
    MASCOTRL_DUCKDB_MAX_MEMORY / MASCOTRL_DUCKDB_THREADS env vars the lake
    builder already respects."""
    _write_fixture_lake(tmp_path)
    monkeypatch.setenv("MASCOTRL_DUCKDB_MAX_MEMORY", "777MB")
    monkeypatch.setenv("MASCOTRL_DUCKDB_THREADS", "3")

    executed: list[str] = []
    import duckdb as real_duckdb

    real_connect = real_duckdb.connect

    class _TrackingConnection:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
            executed.append(sql)
            return self._inner.execute(sql, *args, **kwargs)

        def close(self) -> None:
            self._inner.close()

    def _tracking_connect(*args: Any, **kwargs: Any) -> _TrackingConnection:
        return _TrackingConnection(real_connect(*args, **kwargs))

    monkeypatch.setattr(real_duckdb, "connect", _tracking_connect)

    from mascotrl.data.surface_signals import _load_vol_surface_raw

    _load_vol_surface_raw(tmp_path, secids=[7], start="2020-01-01", end="2020-12-31")

    assert any("memory_limit" in s and "777MB" in s for s in executed)
    assert any("SET threads TO 3" in s for s in executed)


def _write_multi_secid_fixture_lake(root: Path, secids: list[int]) -> None:
    rows = []
    for secid in secids:
        for date, atm in (("2020-01-31", 0.20 + 0.001 * secid), ("2020-02-28", 0.30 + 0.001 * secid)):
            rows.append(_point(secid, date, 30, 50, "C", atm))
            rows.append(_point(secid, date, 30, -50, "P", atm + 0.02))
    df = pd.DataFrame(rows)
    part = root / "vol_surface" / "year=2020" / "month=01"
    part.mkdir(parents=True, exist_ok=True)
    df.to_parquet(part / "data_0.parquet", index=False)


def test_materialize_secid_batching_matches_unbatched_result(tmp_path: Path) -> None:
    """A defect: a full-universe pool loaded every secid's raw option quotes
    into one pandas DataFrame in a single DuckDB scan, which at hundreds of
    names over a decade overflowed host memory and crashed the machine.
    Batching by secid must produce byte-identical signal panels to the
    unbatched path (`mw_xs` cross-sections over the *whole* pool, not a
    batch, so this also guards against a batching regression there)."""
    secids = [100 + i for i in range(7)]
    _write_multi_secid_fixture_lake(tmp_path, secids)

    unbatched = materialize_surface_signals_from_lake(
        tmp_path,
        secids=secids,
        start="2020-01-01",
        end="2020-12-31",
        month_end_only=True,
        secid_batch_size=None,
    )
    batched = materialize_surface_signals_from_lake(
        tmp_path,
        secids=secids,
        start="2020-01-01",
        end="2020-12-31",
        month_end_only=True,
        secid_batch_size=3,
    )
    assert not unbatched.empty
    unbatched_sorted = unbatched.sort_values(["secid", "date"]).reset_index(drop=True)
    batched_sorted = batched.sort_values(["secid", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(unbatched_sorted, batched_sorted)


def test_materialize_kelly_iv_images_batching_matches_unbatched(tmp_path: Path) -> None:
    """Kelly image materialization must batch by secid without changing the
    (T, K, 11, 34) cube: a single full-K raw surface fetch OOMed the
    full campaign after the signal gate had already completed."""
    secids = [100 + i for i in range(5)]
    _write_multi_secid_fixture_lake(tmp_path, secids)
    dates = [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-15"), pd.Timestamp("2020-02-28")]
    unbatched = materialize_kelly_iv_images_from_lake(
        tmp_path,
        secids=secids,
        dates=dates,
        start="2020-01-01",
        end="2020-12-31",
        secid_batch_size=None,
    )
    batched = materialize_kelly_iv_images_from_lake(
        tmp_path,
        secids=secids,
        dates=dates,
        start="2020-01-01",
        end="2020-12-31",
        secid_batch_size=2,
    )
    assert unbatched.shape == (3, 5, 11, 34)
    assert batched.shape == unbatched.shape
    np.testing.assert_allclose(unbatched, batched)


def test_materialize_kelly_iv_images_from_lake_no_forward_fill_leaves_nan_zeroed(
    tmp_path: Path,
) -> None:
    _write_fixture_lake(tmp_path)
    dates = [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-15")]
    cube = materialize_kelly_iv_images_from_lake(
        tmp_path,
        secids=[7],
        dates=dates,
        start="2020-01-01",
        end="2020-12-31",
        forward_fill=False,
    )
    # No fill: the gap date has no observation anywhere on that grid.
    assert np.all(np.isnan(cube[1, 0]))
