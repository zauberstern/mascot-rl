"""Phase A: Parquet Data Lake builder via DuckDB (memory-safe, no pandas.read_csv).

Large OptionMetrics CSVs (10–180GB) MUST NOT be loaded whole into RAM.

Strategy for huge files (>= MASCOTRL_HUGE_CSV_GB, default 2GB):
  1. Stream-split the CSV into per-year staging files (one row in RAM at a time)
  2. Convert each year file with DuckDB under a hard memory_limit (default 8GB),
     spilling to temp_directory on the lake disk

Small files: single DuckDB COPY with the same memory/temp limits.
Never use sample_size=-1 on huge inputs (that OOM-killed the host previously).
"""
from __future__ import annotations

import csv
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import duckdb

from mascotrl.data.paths import LAKE_ROOT, TIER_A, TIER_B, ensure_lake_dirs, tier_a_available
from mascotrl.logging_utils import get_logger

log = get_logger("volsurf.l5.lake")

DEFAULT_MAX_MEMORY = os.environ.get("MASCOTRL_DUCKDB_MAX_MEMORY", "8GB")
DEFAULT_THREADS = int(os.environ.get("MASCOTRL_DUCKDB_THREADS", "2"))
HUGE_GB_THRESHOLD = float(os.environ.get("MASCOTRL_HUGE_CSV_GB", "2.0"))
YEAR_START = int(os.environ.get("MASCOTRL_LAKE_YEAR_START", "2003"))
YEAR_END = int(os.environ.get("MASCOTRL_LAKE_YEAR_END", "2024"))
# Log every N rows during stream split so progress is visible without buffering forever
SPLIT_LOG_EVERY = int(os.environ.get("MASCOTRL_SPLIT_LOG_EVERY", "5_000_000"))


class ParquetDataLakeBuilder:
    """Run-once EL: CSV → ZSTD Hive-partitioned Parquet, RAM-capped."""

    DATASET_STAGING_STEM = {
        "options_panel": Path(TIER_A["options_panel"]).stem,
        "vol_surface": Path(TIER_A["vol_surface"]).stem,
    }

    def __init__(
        self,
        lake_base_dir: str | Path | None = None,
        max_memory: str = DEFAULT_MAX_MEMORY,
        threads: int = DEFAULT_THREADS,
        huge_gb_threshold: float = HUGE_GB_THRESHOLD,
    ):
        # Explicit lake path (tests / overrides): skip USB mount preflight.
        if lake_base_dir is None:
            ensure_lake_dirs()
            self.lake_base_dir = Path(LAKE_ROOT)
        else:
            self.lake_base_dir = Path(lake_base_dir)
        self.lake_base_dir.mkdir(parents=True, exist_ok=True)
        self.huge_gb_threshold = huge_gb_threshold
        self.max_memory = max_memory
        self.threads = max(1, threads)

        self.temp_dir = self.lake_base_dir / "_duckdb_tmp"
        self.staging_dir = self.lake_base_dir / "_csv_year_staging"
        self.rejects_dir = self.lake_base_dir / "_ingest_rejects"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.rejects_dir.mkdir(parents=True, exist_ok=True)

        db_path = self.lake_base_dir / "_duckdb_catalog.db"
        log.info(
            "ParquetDataLakeBuilder lake=%s max_memory=%s threads=%d temp=%s",
            self.lake_base_dir,
            self.max_memory,
            self.threads,
            self.temp_dir,
        )
        self.con = duckdb.connect(database=str(db_path))
        self._apply_resource_limits()

    def _apply_resource_limits(self) -> None:
        self.con.execute("SET preserve_insertion_order = false;")
        self.con.execute(f"SET temp_directory = '{self.temp_dir.as_posix()}';")
        self.con.execute("SET max_temp_directory_size = '400GB';")
        self.con.execute(f"SET memory_limit = '{self.max_memory}';")
        self.con.execute(f"SET max_memory = '{self.max_memory}';")
        self.con.execute(f"SET threads = {self.threads};")
        for stmt in (
            "SET enable_external_file_cache = false;",
            "SET force_external = true;",
        ):
            try:
                self.con.execute(stmt)
            except Exception:
                pass
        log.info(
            "DuckDB limits memory_limit=%s threads=%d spill=%s",
            self.max_memory,
            self.threads,
            self.temp_dir,
        )

    def _csv_read_opts(self, *, huge: bool, all_varchar: bool = False) -> str:
        sample = 10_000 if huge else 50_000
        parts = [
            f"sample_size={sample}",
            "ignore_errors=true",
            "strict_mode=false",
            "null_padding=true",
            "parallel=false",
            "max_line_size=10000000",
        ]
        if all_varchar:
            parts.append("all_varchar=true")
        return ", ".join(parts)

    def _detect_date_col(self, src: Path, *, huge: bool) -> str | None:
        opts = self._csv_read_opts(huge=huge)
        cols = self.con.execute(
            f"DESCRIBE SELECT * FROM read_csv_auto('{src.as_posix()}', {opts})"
        ).fetchall()
        colnames = {c[0].lower(): c[0] for c in cols}
        for cand in ("date", "date_dt", "datadate", "trade_date", "observation_date"):
            if cand in colnames:
                return colnames[cand]
        return None

    @staticmethod
    def _parse_year(value: str) -> int | None:
        if not value:
            return None
        v = value.strip().strip('"')
        # ISO / OptionMetrics: YYYY-MM-DD or YYYYMMDD
        if len(v) >= 4 and v[:4].isdigit():
            y = int(v[:4])
            if YEAR_START <= y <= YEAR_END:
                return y
        return None

    def _stream_split_by_year(self, src: Path, date_col: str) -> dict[int, Path]:
        """One-pass CSV split → staging/year=YYYY.csv (constant memory)."""
        dest_dir = self.staging_dir / src.stem
        dest_dir.mkdir(parents=True, exist_ok=True)
        existing = {
            int(p.stem.split("=")[1]): p
            for p in dest_dir.glob("year=*.csv")
            if p.stat().st_size > 0
        }
        if len(existing) >= (YEAR_END - YEAR_START + 1):
            log.info(
                "Reusing %d staged year CSVs under %s",
                len(existing),
                dest_dir,
            )
            return existing

        log.info(
            "Stream-splitting %s by year → %s (row-at-a-time, low RAM)",
            src.name,
            dest_dir,
        )
        t0 = time.perf_counter()
        writers: dict[int, csv.writer] = {}
        handles: dict[int, object] = {}
        paths: dict[int, Path] = {}
        header: list[str] | None = None
        date_idx = -1
        n_rows = 0
        n_kept = 0
        n_skip = 0

        try:
            with open(src, "r", newline="", encoding="utf-8", errors="replace") as fin:
                reader = csv.reader(fin)
                header = next(reader)
                lower = [h.strip().lower() for h in header]
                try:
                    date_idx = lower.index(date_col.lower())
                except ValueError as exc:
                    raise RuntimeError(
                        f"date column {date_col!r} not in header of {src}"
                    ) from exc

                for row in reader:
                    n_rows += 1
                    if n_rows % SPLIT_LOG_EVERY == 0:
                        log.info(
                            "split progress rows=%d kept=%d skip=%d open_years=%d elapsed=%.0fs",
                            n_rows,
                            n_kept,
                            n_skip,
                            len(paths),
                            time.perf_counter() - t0,
                        )
                    if date_idx >= len(row):
                        n_skip += 1
                        continue
                    year = self._parse_year(row[date_idx])
                    if year is None:
                        n_skip += 1
                        continue
                    if year not in writers:
                        path = dest_dir / f"year={year}.csv"
                        fh = open(path, "w", newline="", encoding="utf-8")
                        w = csv.writer(fh)
                        w.writerow(header)
                        handles[year] = fh
                        writers[year] = w
                        paths[year] = path
                        log.info("Opened staging %s", path.name)
                    writers[year].writerow(row)
                    n_kept += 1
        finally:
            for fh in handles.values():
                fh.close()

        log.info(
            "Stream-split done rows=%d kept=%d skip=%d years=%d in %.1fs",
            n_rows,
            n_kept,
            n_skip,
            len(paths),
            time.perf_counter() - t0,
        )
        return paths

    def _convert_year_csv_to_parquet(
        self, year_csv: Path, out: Path, year: int, date_col: str
    ) -> None:
        year_dir = out / f"year={year}"
        if year_dir.is_dir() and any(year_dir.rglob("*.parquet")):
            log.info("Skip parquet year=%d (exists)", year)
            return
        opts = self._csv_read_opts(huge=False)  # per-year file is much smaller
        t0 = time.perf_counter()
        size_gb = year_csv.stat().st_size / 1e9
        log.info(
            "DuckDB convert year=%d file=%.2fGB → %s (cap %s)",
            year,
            size_gb,
            out.name,
            self.max_memory,
        )
        sql = f"""
        COPY (
            SELECT
                *,
                {year} AS year,
                EXTRACT(month FROM TRY_CAST("{date_col}" AS DATE))::INTEGER AS month
            FROM read_csv_auto('{year_csv.as_posix()}', {opts})
            WHERE TRY_CAST("{date_col}" AS DATE) IS NOT NULL
        ) TO '{out.as_posix()}'
        (FORMAT PARQUET, PARTITION_BY (year, month), COMPRESSION 'ZSTD',
         ROW_GROUP_SIZE 122880, OVERWRITE_OR_IGNORE true);
        """
        try:
            self.con.execute(sql)
            try:
                self.con.execute("CHECKPOINT;")
            except Exception:
                pass
            log.info("year=%d parquet done in %.1fs", year, time.perf_counter() - t0)
        except Exception as exc:
            log.error("year=%d convert failed: %s", year, exc)
            self._apply_resource_limits()
            raise

    def _execute_huge_conversion(self, src: Path, out: Path, date_col: str) -> None:
        staged = self._stream_split_by_year(src, date_col)
        for year in sorted(staged):
            self._convert_year_csv_to_parquet(staged[year], out, year, date_col)

    def _execute_small_partitioned(
        self, src: Path, out: Path, date_col: str
    ) -> None:
        opts = self._csv_read_opts(huge=False)
        t0 = time.perf_counter()
        self.con.execute(
            f"""
            COPY (
                SELECT *,
                    EXTRACT(year FROM CAST("{date_col}" AS DATE))::INTEGER AS year,
                    EXTRACT(month FROM CAST("{date_col}" AS DATE))::INTEGER AS month
                FROM read_csv_auto('{src.as_posix()}', {opts})
            ) TO '{out.as_posix()}'
            (FORMAT PARQUET, PARTITION_BY (year, month), COMPRESSION 'ZSTD',
             ROW_GROUP_SIZE 122880, OVERWRITE_OR_IGNORE true);
            """
        )
        log.info("Partitioned write %s in %.1fs", out.name, time.perf_counter() - t0)

    def _execute_partitioned_conversion(self, source_csv: str, target_table: str) -> None:
        src = Path(source_csv)
        if not src.exists():
            log.warning("Skipping missing source: %s", src)
            return
        out = self.lake_base_dir / target_table
        out.mkdir(parents=True, exist_ok=True)
        size_gb = src.stat().st_size / 1e9
        huge = size_gb >= self.huge_gb_threshold
        log.info(
            "Converting %s (%.2f GB) → %s mode=%s",
            src.name,
            size_gb,
            out,
            "stream_split+yearly_duckdb" if huge else "single_pass",
        )
        t0 = time.perf_counter()
        date_col = self._detect_date_col(src, huge=huge)
        if date_col is None:
            dest = out / "data.parquet"
            if dest.exists() and dest.stat().st_size > 0:
                log.info("Skip existing %s", dest)
                return
            opts = self._csv_read_opts(huge=huge)
            self.con.execute(
                f"""
                COPY (
                    SELECT * FROM read_csv_auto('{src.as_posix()}', {opts})
                ) TO '{dest.as_posix()}'
                (FORMAT PARQUET, COMPRESSION 'ZSTD', ROW_GROUP_SIZE 122880);
                """
            )
            log.info("Wrote non-partitioned %s in %.1fs", dest, time.perf_counter() - t0)
            return

        if huge:
            self._execute_huge_conversion(src, out, date_col)
        else:
            self._execute_small_partitioned(src, out, date_col)

        log.info("Finished %s → %s in %.1fs", src.name, out, time.perf_counter() - t0)

    @staticmethod
    def _schema_column_names(con: duckdb.DuckDBPyConnection, files: Sequence[str]) -> set[str]:
        if not files:
            return set()
        listed = ", ".join(f"'{f}'" for f in files)
        rows = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet([{listed}], union_by_name=true) LIMIT 0"
        ).fetchall()
        return {str(r[0]) for r in rows}

    def _count_parquet_year(self, year_dir: Path) -> int:
        files = [p.as_posix() for p in year_dir.rglob("*.parquet")] if year_dir.is_dir() else []
        if not files:
            return 0
        listed = ", ".join(f"'{f}'" for f in files)
        return int(
            self.con.execute(
                f"SELECT COUNT(*) FROM read_parquet([{listed}], union_by_name=true)"
            ).fetchone()[0]
        )

    def _write_year_rejects(
        self,
        *,
        dataset: str,
        year: int,
        year_csv: Path,
        date_col: str,
    ) -> tuple[Path, int]:
        """Persist undateable rows under ``_ingest_rejects`` (off hive globs)."""
        out_dir = self.rejects_dir / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"year={year}.parquet"
        opts = self._csv_read_opts(huge=False, all_varchar=True)
        self.con.execute(
            f"""
            COPY (
                SELECT *,
                       'bad_or_null_date' AS reject_reason
                FROM read_csv_auto('{year_csv.as_posix()}', {opts})
                WHERE TRY_CAST("{date_col}" AS DATE) IS NULL
            ) TO '{dest.as_posix()}'
            (FORMAT PARQUET, COMPRESSION 'ZSTD', OVERWRITE_OR_IGNORE true);
            """
        )
        n = int(
            self.con.execute(
                f"SELECT COUNT(*) FROM read_parquet('{dest.as_posix()}')"
            ).fetchone()[0]
        )
        if n == 0 and dest.is_file():
            # Keep empty marker file for seal accounting (explicit zero rejects).
            pass
        return dest, n

    def _convert_year_csv_to_tmp(
        self,
        year_csv: Path,
        tmp_root: Path,
        year: int,
        date_col: str,
        *,
        prior_types: dict[str, str] | None = None,
    ) -> None:
        """Write hive partitions under ``tmp_root`` (caller supplies empty dir).

        Reads CSV as all-varchar then TRY_CASTs using prior parquet types so
        DuckDB typed inference cannot silently drop rows (ignore_errors).
        """
        opts = self._csv_read_opts(huge=False, all_varchar=True)
        size_gb = year_csv.stat().st_size / 1e9
        log.info(
            "DuckDB force-convert year=%d file=%.2fGB → tmp %s (cap %s)",
            year,
            size_gb,
            tmp_root,
            self.max_memory,
        )
        t0 = time.perf_counter()
        # Column list from CSV header
        with year_csv.open("r", encoding="utf-8", errors="replace") as f:
            header = next(csv.reader(f))
        prior_types = {k.lower(): v for k, v in (prior_types or {}).items()}

        def sql_type_for(col: str) -> str:
            if col.lower() == date_col.lower():
                return "DATE"
            t = prior_types.get(col.lower(), "")
            tl = t.upper()
            if "DATE" in tl or "TIMESTAMP" in tl:
                return "DATE"
            if any(x in tl for x in ("DOUBLE", "FLOAT", "REAL", "DECIMAL", "HUGEINT")):
                return "DOUBLE"
            if any(x in tl for x in ("BIGINT", "HUGEINT", "UBIGINT")):
                return "BIGINT"
            if "INT" in tl:
                return "INTEGER"
            if "BOOL" in tl:
                return "BOOLEAN"
            return "VARCHAR"

        select_parts: list[str] = []
        for col in header:
            if col.lower() in ("year", "month"):
                continue
            st = sql_type_for(col)
            select_parts.append(f'TRY_CAST("{col}" AS {st}) AS "{col}"')
        select_parts.append(f"{year} AS year")
        select_parts.append(
            f'EXTRACT(month FROM TRY_CAST("{date_col}" AS DATE))::INTEGER AS month'
        )
        select_sql = ",\n                ".join(select_parts)
        sql = f"""
        COPY (
            SELECT
                {select_sql}
            FROM read_csv_auto('{year_csv.as_posix()}', {opts})
            WHERE TRY_CAST("{date_col}" AS DATE) IS NOT NULL
        ) TO '{tmp_root.as_posix()}'
        (FORMAT PARQUET, PARTITION_BY (year, month), COMPRESSION 'ZSTD',
         ROW_GROUP_SIZE 122880, OVERWRITE_OR_IGNORE true);
        """
        self.con.execute(sql)
        try:
            self.con.execute("CHECKPOINT;")
        except Exception:
            pass
        log.info("year=%d tmp parquet done in %.1fs", year, time.perf_counter() - t0)

    @staticmethod
    def _schema_types(con: duckdb.DuckDBPyConnection, files: Sequence[str]) -> dict[str, str]:
        if not files:
            return {}
        listed = ", ".join(f"'{f}'" for f in files)
        rows = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet([{listed}], union_by_name=true) LIMIT 0"
        ).fetchall()
        return {str(r[0]): str(r[1]) for r in rows}

    def reconvert_years(
        self,
        dataset: str,
        years: Iterable[int],
        *,
        force: bool = True,
        backup: bool = True,
        date_col: str | None = None,
    ) -> dict[str, Any]:
        """Force re-parquet selected years from staging; rejects off hive path.

        Preserves column names vs prior year schema; refuses row shrinkage.
        """
        if dataset not in self.DATASET_STAGING_STEM:
            raise ValueError(f"unsupported dataset {dataset!r}")
        if not force:
            raise ValueError("reconvert_years requires force=True")

        stem = self.DATASET_STAGING_STEM[dataset]
        out_root = self.lake_base_dir / dataset
        out_root.mkdir(parents=True, exist_ok=True)
        year_list = sorted({int(y) for y in years})
        details: dict[int, dict[str, Any]] = {}

        for year in year_list:
            log.info("reconvert start dataset=%s year=%d", dataset, year)
            year_csv = self.staging_dir / stem / f"year={year}.csv"
            if not year_csv.is_file():
                details[year] = {"pass": False, "reason": "staging_missing", "path": str(year_csv)}
                continue

            year_dir = out_root / f"year={year}"
            prior_files = (
                [p.as_posix() for p in year_dir.rglob("*.parquet")] if year_dir.is_dir() else []
            )
            prior_cols = self._schema_column_names(self.con, prior_files)
            prior_types = self._schema_types(self.con, prior_files)
            prior_n = self._count_parquet_year(year_dir)

            dc = date_col or self._detect_date_col(year_csv, huge=False)
            if not dc:
                details[year] = {"pass": False, "reason": "date_col_missing"}
                continue

            tmp_root = out_root / f".reconvert_tmp_{year}"
            if tmp_root.exists():
                shutil.rmtree(tmp_root)
            tmp_root.mkdir(parents=True, exist_ok=True)

            try:
                self._convert_year_csv_to_tmp(
                    year_csv, tmp_root, year, dc, prior_types=prior_types
                )
                # DuckDB PARTITION_BY writes year=Y under tmp_root
                produced = tmp_root / f"year={year}"
                if not produced.is_dir() or not any(produced.rglob("*.parquet")):
                    raise RuntimeError(f"no parquet produced under {produced}")

                new_files = [p.as_posix() for p in produced.rglob("*.parquet")]
                new_cols = self._schema_column_names(self.con, new_files)
                # Ignore accidental hive keys from temp path segments
                new_cols = {c for c in new_cols if not c.startswith("_") and c != "reconvert_tmp"}
                # Partition cols year/month are additive; require prior cols ⊆ new
                if prior_cols and not prior_cols.issubset(new_cols | {"year", "month"}):
                    missing = sorted(prior_cols - new_cols - {"year", "month"})
                    raise RuntimeError(f"schema freeze violated; missing columns {missing}")

                new_n = self._count_parquet_year(produced)
                if new_n < prior_n:
                    raise RuntimeError(
                        f"row shrinkage refused: prior={prior_n} new={new_n} year={year}"
                    )

                reject_path, reject_n = self._write_year_rejects(
                    dataset=dataset, year=year, year_csv=year_csv, date_col=dc
                )

                from mascotrl.data.lake_source_audit import line_count as _line_count

                staging_n = max(0, _line_count(year_csv) - 1)
                if new_n + reject_n != staging_n:
                    raise RuntimeError(
                        f"staging accounting failed: staging={staging_n} "
                        f"parquet={new_n} rejects={reject_n} year={year}"
                    )

                bak = out_root / f"year={year}.bak"
                if year_dir.exists():
                    if backup:
                        if bak.exists():
                            shutil.rmtree(bak)
                        year_dir.rename(bak)
                    else:
                        shutil.rmtree(year_dir)
                produced.rename(year_dir)
                # Clean empty tmp_root
                if tmp_root.exists():
                    shutil.rmtree(tmp_root, ignore_errors=True)
                if backup and bak.exists():
                    shutil.rmtree(bak)

                details[year] = {
                    "pass": True,
                    "prior_parquet_rows": prior_n,
                    "parquet_rows": new_n,
                    "reject_rows": reject_n,
                    "reject_path": str(reject_path),
                    "prior_columns": sorted(prior_cols),
                    "new_columns": sorted(new_cols),
                }
                log.info(
                    "reconvert done dataset=%s year=%d parquet=%d rejects=%d",
                    dataset,
                    year,
                    new_n,
                    reject_n,
                )
            except Exception as exc:
                log.error("reconvert failed dataset=%s year=%d: %s", dataset, year, exc)
                if tmp_root.exists():
                    shutil.rmtree(tmp_root, ignore_errors=True)
                details[year] = {
                    "pass": False,
                    "reason": str(exc),
                    "prior_parquet_rows": prior_n,
                }
                self._apply_resource_limits()

        return {
            "dataset": dataset,
            "years": details,
            "pass": all(d.get("pass") for d in details.values()) if details else False,
        }

    def build_options_panel_lake(self) -> None:
        out = self.lake_base_dir / "options_panel"
        years_present = {p.name for p in out.glob("year=*") if p.is_dir()}
        if len(years_present) >= 10:
            log.info(
                "Skipping options_panel — %d year partitions already under %s",
                len(years_present),
                out,
            )
            return
        self._execute_partitioned_conversion(str(TIER_A["options_panel"]), "options_panel")

    def build_vol_surface_lake(self) -> None:
        primary = TIER_A["vol_surface"]
        if not primary.exists():
            log.warning("Primary vol surface missing: %s", primary)
            return
        out = self.lake_base_dir / "vol_surface"
        out.mkdir(parents=True, exist_ok=True)
        years_present = {p.name for p in out.glob("year=*") if p.is_dir()}
        if len(years_present) >= 10:
            log.info(
                "Skipping vol_surface — %d year partitions already under %s",
                len(years_present),
                out,
            )
            return
        log.info(
            "Converting vol_surface %s (%.2f GB) …",
            primary.name,
            primary.stat().st_size / 1e9,
        )
        self._execute_partitioned_conversion(str(primary), "vol_surface")

    def build_macro_features_lake(self) -> None:
        macro_dir = self.lake_base_dir / "macro"
        macro_dir.mkdir(parents=True, exist_ok=True)
        for name, path in TIER_B.items():
            if not path.exists():
                log.warning("Tier B missing: %s", path)
                continue
            dest = macro_dir / f"{name}.parquet"
            if dest.exists() and dest.stat().st_size > 0:
                log.info("Skipping existing macro %s", dest.name)
                continue
            size_gb = path.stat().st_size / 1e9
            huge = size_gb >= self.huge_gb_threshold
            if huge:
                # Large macro (e.g. SPX dump): same stream-split path if dated
                date_col = self._detect_date_col(path, huge=True)
                if date_col:
                    log.info(
                        "Macro %s is large (%.2f GB) — stream-split by year",
                        path.name,
                        size_gb,
                    )
                    sub = macro_dir / name
                    sub.mkdir(exist_ok=True)
                    self._execute_huge_conversion(path, sub, date_col)
                    continue
            opts = self._csv_read_opts(huge=huge, all_varchar=True)
            try:
                self.con.execute(
                    f"""
                    COPY (
                        SELECT * FROM read_csv_auto('{path.as_posix()}', {opts})
                    ) TO '{dest.as_posix()}'
                    (FORMAT PARQUET, COMPRESSION 'ZSTD', ROW_GROUP_SIZE 122880);
                    """
                )
                log.info("Wrote %s", dest)
            except Exception as exc:
                log.warning("Failed converting %s (%s); skipping", path.name, exc)

    def execute_full_build(self, include_tier_a: bool = True) -> None:
        log.info(
            "execute_full_build lake=%s include_tier_a=%s max_memory=%s",
            self.lake_base_dir,
            include_tier_a,
            self.max_memory,
        )
        if include_tier_a:
            if not tier_a_available():
                raise FileNotFoundError(
                    "Tier A OptionMetrics mount not available at "
                    f"{TIER_A['options_panel'].parent}"
                )
            self.build_options_panel_lake()
            self.build_vol_surface_lake()
        self.build_macro_features_lake()
        log.info("execute_full_build finished")
        log.info("Data lake build complete")
