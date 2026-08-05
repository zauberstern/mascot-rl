"""Shared fingerprint-keyed surface cache (MASCOTRL_SURFACE_CACHE_DIR)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def test_surface_fingerprint_stable_under_secid_reorder() -> None:
    from mascotrl.data.surface_signals import surface_signals_cache_fingerprint

    a = surface_signals_cache_fingerprint(
        secids=[3, 1, 2], start="2020-01-01", end="2020-12-31"
    )
    b = surface_signals_cache_fingerprint(
        secids=[1, 2, 3], start="2020-01-01", end="2020-12-31"
    )
    assert a == b
    assert len(a) == 64


def test_shared_surface_cache_hit_skips_duckdb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mascotrl.data import surface_signals as ss

    cache_dir = tmp_path / "shared"
    monkeypatch.setenv("MASCOTRL_SURFACE_CACHE_DIR", str(cache_dir))

    panel = pd.DataFrame(
        {
            "secid": [7, 7],
            "date": pd.to_datetime(["2020-01-31", "2020-02-28"]),
            "iv_skew_30d": [0.1, 0.2],
        }
    )

    calls = {"n": 0}

    def _fake_raw(*_a, **_k):
        calls["n"] += 1
        return pd.DataFrame(
            columns=[
                "secid",
                "date",
                "days",
                "delta",
                "cp_flag",
                "impl_volatility",
                "impl_strike",
                "impl_premium",
                "dispersion",
            ]
        )

    def _fake_panel(*_a, **_k):
        return panel.copy()

    monkeypatch.setattr(ss, "_load_vol_surface_raw", _fake_raw)
    monkeypatch.setattr(ss, "compute_surface_signals_panel", _fake_panel)

    first = ss.materialize_surface_signals_from_lake(
        "/unused",
        secids=[7],
        start="2020-01-01",
        end="2020-12-31",
    )
    assert calls["n"] == 1
    assert len(list(cache_dir.glob("*.parquet"))) == 1

    def _boom(*_a, **_k):
        calls["n"] += 1
        raise AssertionError("DuckDB reload on shared surface cache hit")

    monkeypatch.setattr(ss, "_load_vol_surface_raw", _boom)
    second = ss.materialize_surface_signals_from_lake(
        "/unused",
        secids=[7],
        start="2020-01-01",
        end="2020-12-31",
    )
    assert calls["n"] == 1
    pd.testing.assert_frame_equal(
        first.reset_index(drop=True), second.reset_index(drop=True)
    )


def test_shared_surface_cache_corrupt_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mascotrl.data.surface_signals import (
        materialize_surface_signals_from_lake,
        surface_signals_cache_fingerprint,
    )

    cache_dir = tmp_path / "shared"
    cache_dir.mkdir()
    monkeypatch.setenv("MASCOTRL_SURFACE_CACHE_DIR", str(cache_dir))
    fp = surface_signals_cache_fingerprint(
        secids=[7], start="2020-01-01", end="2020-12-31"
    )
    bad = cache_dir / f"{fp}.parquet"
    bad.write_text("not-a-parquet", encoding="utf-8")
    (cache_dir / f"{fp}.meta.json").write_text(
        f'{{"fingerprint": "{fp}"}}\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="surface cache corrupt"):
        materialize_surface_signals_from_lake(
            "/unused",
            secids=[7],
            start="2020-01-01",
            end="2020-12-31",
        )


def test_duckdb_threads_default_is_four(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset MASCOTRL_DUCKDB_THREADS must default to 4 (not 2)."""
    import duckdb as real_duckdb

    from mascotrl.data.surface_signals import _load_vol_surface_raw

    part = tmp_path / "vol_surface" / "year=2020" / "month=01"
    part.mkdir(parents=True)
    pd.DataFrame(
        {
            "secid": [7],
            "date": [pd.Timestamp("2020-01-31")],
            "days": [30],
            "delta": [50],
            "cp_flag": ["C"],
            "impl_volatility": [0.2],
            "impl_strike": [100.0],
            "impl_premium": [1.0],
            "dispersion": [0.01],
        }
    ).to_parquet(part / "data_0.parquet", index=False)

    monkeypatch.delenv("MASCOTRL_DUCKDB_THREADS", raising=False)
    executed: list[str] = []
    real_connect = real_duckdb.connect

    class _Tracking:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql: str, *a, **k):
            executed.append(sql)
            return self._inner.execute(sql, *a, **k)

        def close(self):
            self._inner.close()

    monkeypatch.setattr(
        real_duckdb, "connect", lambda *a, **k: _Tracking(real_connect(*a, **k))
    )
    _load_vol_surface_raw(tmp_path, secids=[7], start="2020-01-01", end="2020-12-31")
    assert any("SET threads TO 4" in s for s in executed)
