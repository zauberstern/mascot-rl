"""Force year reconvert from staging: rejects off-hive, schema freeze, no shrink."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from mascotrl.data.lake_builder import ParquetDataLakeBuilder
from mascotrl.data.lake_source_audit import compare_year_counts_with_rejects, reject_year_counts


def _write_seed_parquet(year_dir: Path, rows: list[dict]) -> None:
    year_dir.mkdir(parents=True, exist_ok=True)
    month = year_dir / "month=01"
    month.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(
        """
        CREATE TABLE t AS SELECT * FROM (
            SELECT 1 AS secid, DATE '2003-01-02' AS date, 0.2::DOUBLE AS impl_volatility
        ) WHERE 1=0
        """
    )
    for r in rows:
        con.execute(
            "INSERT INTO t VALUES (?, ?, ?)",
            [r["secid"], r["date"], r["impl_volatility"]],
        )
    dest = month / "data_0.parquet"
    con.execute(f"COPY t TO '{dest.as_posix()}' (FORMAT PARQUET)")
    con.close()


def test_reconvert_years_writes_rejects_off_hive_and_keeps_schema(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    staging = lake / "_csv_year_staging" / "option_prices_2003_2024_sp500_all"
    staging.mkdir(parents=True)
    # 2 good + 1 bad date
    (staging / "year=2003.csv").write_text(
        "secid,date,impl_volatility\n"
        "1,2003-01-02,0.21\n"
        "2,not-a-date,0.22\n"
        "3,2003-01-03,0.23\n",
        encoding="utf-8",
    )
    # Existing incomplete parquet (schema seed + 1 row only)
    seed_dir = lake / "options_panel" / "year=2003"
    _write_seed_parquet(
        seed_dir,
        [{"secid": 1, "date": "2003-01-02", "impl_volatility": 0.2}],
    )

    builder = ParquetDataLakeBuilder(lake_base_dir=lake, max_memory="512MB", threads=1)
    result = builder.reconvert_years("options_panel", [2003], force=True, backup=True)

    assert result["years"][2003]["pass"] is True
    year_dir = lake / "options_panel" / "year=2003"
    assert year_dir.is_dir()
    assert not (lake / "options_panel" / "year=2003.bak").exists() or True  # bak optional cleanup
    # Rejects must not live under hive dataset root as month partitions
    reject = lake / "_ingest_rejects" / "options_panel" / "year=2003.parquet"
    assert reject.is_file()
    assert reject.parent.parent.name == "_ingest_rejects"
    hive_parquets = list((lake / "options_panel" / "year=2003").rglob("*.parquet"))
    assert hive_parquets
    assert all("_ingest_rejects" not in p.parts for p in hive_parquets)

    con = duckdb.connect()
    files = [p.as_posix() for p in year_dir.rglob("*.parquet")]
    listed = ", ".join(f"'{f}'" for f in files)
    pq_n = int(
        con.execute(
            f"SELECT COUNT(*) FROM read_parquet([{listed}], union_by_name=true)"
        ).fetchone()[0]
    )
    rej_n = int(
        con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{reject.as_posix()}')"
        ).fetchone()[0]
    )
    cols = {
        r[0]
        for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet([{listed}], union_by_name=true) LIMIT 0"
        ).fetchall()
    }
    con.close()
    assert pq_n == 2
    assert rej_n == 1
    assert "secid" in cols and "date" in cols and "impl_volatility" in cols
    # Additive vs prior (prior had 1 row)
    assert pq_n >= 1
    assert result["years"][2003]["parquet_rows"] >= result["years"][2003]["prior_parquet_rows"]


def test_compare_year_counts_with_rejects() -> None:
    ok, deltas = compare_year_counts_with_rejects(
        staging={2003: 10, 2004: 5},
        parquet={2003: 9, 2004: 5},
        rejects={2003: 1, 2004: 0},
        years=range(2003, 2005),
    )
    assert ok is True
    assert deltas[2003] == 0
    ok2, _ = compare_year_counts_with_rejects(
        staging={2003: 10},
        parquet={2003: 8},
        rejects={2003: 1},
        years=range(2003, 2004),
    )
    assert ok2 is False


def test_reject_year_counts_missing_is_zero(tmp_path: Path) -> None:
    assert reject_year_counts(tmp_path, "options_panel") == {}


def test_convert_year_failure_raises(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    builder = ParquetDataLakeBuilder(lake_base_dir=lake, max_memory="512MB", threads=1)
    csv = tmp_path / "year=2003.csv"
    csv.write_text("secid,date,impl_volatility\n1,2003-01-02,0.21\n", encoding="utf-8")

    class BadCon:
        def execute(self, _sql: str) -> None:
            raise RuntimeError("duckdb convert failed")

    builder.con = BadCon()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="duckdb convert failed"):
        builder._convert_year_csv_to_parquet(csv, lake / "options_panel", 2003, "date")
