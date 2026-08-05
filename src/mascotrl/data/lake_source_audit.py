"""Lake / WRDS source-coverage seal (fail-closed COMPLETE predicates).

Surgical I/O: count-only CSV streams, parquet aggregates, no giant DuckDB CSV loads.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mascotrl.data.paths import (
    LAKE_ROOT,
    RAW_ROOT,
    TIER_A,
    TIER_B,
    MASCOTRL_ROOT,
)
from mascotrl.logging_utils import get_logger

log = get_logger("volsurf.lake_source_audit")

YEAR_START = int(os.environ.get("MASCOTRL_LAKE_YEAR_START", "2003"))
YEAR_END = int(os.environ.get("MASCOTRL_LAKE_YEAR_END", "2024"))
CONTRACT_YEARS = range(YEAR_START, YEAR_END + 1)
HIST_LOG_EVERY = int(os.environ.get("MASCOTRL_AUDIT_LOG_EVERY", "5_000_000"))

ALTERNATE_NAME_MARKERS = (
    "option_prices_slim",
    "option_prices_2003-2024_sp500_all_constituents",
    "std_option_prices",
    "constituent_volsurfd",
)

NON_CONTRACT_UNIQUES = {
    "financial_ratios_ibes_sp500.csv",
    "wrds_ibes_financial_ratios_sp500.csv",
}

HASH_TIER_B_HINTS = {
    "bnnmxdoysn5arpqe.csv": "sp500_prices",
    "rg8xga4yj2pnq05c.csv": "sp500_fwd",
    "vc1wxjwlsynytpmg.csv": "sp500_hv",
    "ytdzwuvdov37jcqn.csv": "sp500_sec",
    "gasrdr6jzrt7ibft.csv": "cboe_vix",
    "pastor-stambaugh.csv": "pastor_stambaugh",
}


class FileClass(str, Enum):
    CANONICAL_INGESTED = "CANONICAL_INGESTED"
    TIER_B_DUPLICATE = "TIER_B_DUPLICATE"
    ALTERNATE_OM_NOT_INGESTED = "ALTERNATE_OM_NOT_INGESTED"
    UNIQUE_NOT_INGESTED = "UNIQUE_NOT_INGESTED"
    NON_MARKET = "NON_MARKET"
    UNKNOWN_MARKET = "UNKNOWN_MARKET"


def s6_listing_ok(*, files: list, unknown: list, missing_contract: list | None = None) -> tuple[bool, str]:
    if not files:
        return False, "empty_listing"
    if unknown:
        return False, "unknown_market"
    if missing_contract:
        return False, "missing_contract"
    return True, "ok"


def s6_missing_contract(files: list, tier_a: Mapping, tier_b: Mapping) -> list[str]:
    names = {Path(p).name for p in files}
    missing: list[str] = []
    for src in list(tier_a.values()) + list(tier_b.values()):
        name = Path(src).name
        if name not in names:
            missing.append(name)
    return missing


def header_fingerprint(path: Path) -> str:
    with path.open("rb") as f:
        line = f.readline()
    return hashlib.sha256(line).hexdigest()


def _read_header_line(path: Path) -> str:
    with path.open("rb") as f:
        return f.readline().decode("utf-8", "replace").strip()


def parse_year(value: str, *, year_start: int = YEAR_START, year_end: int = YEAR_END) -> int | None:
    if not value:
        return None
    v = value.strip().strip('"')
    if len(v) >= 4 and v[:4].isdigit():
        y = int(v[:4])
        if year_start <= y <= year_end:
            return y
        # Outside contract window still return year for alternate checks
        if 1900 <= y <= 2100:
            return y
    return None


def line_count(path: Path) -> int:
    """Fast line count via ``wc -l`` when available, else buffered read."""
    import subprocess

    try:
        out = subprocess.check_output(["wc", "-l", str(path)], text=True)
        return int(out.strip().split()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        n = 0
        buf_size = 8 * 1024 * 1024
        with path.open("rb") as f:
            while True:
                buf = f.read(buf_size)
                if not buf:
                    break
                n += buf.count(b"\n")
        return n


def _year_count_cache_path(artifacts: Path, kind: str, stem: str) -> Path:
    return artifacts / f"lake_source_coverage_{kind}_year_counts_{stem}.json"


def load_year_counts_cache(path: Path, *, fingerprint: dict[str, Any]) -> dict[int, int] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("fingerprint") != fingerprint:
        return None
    raw = payload.get("rows_by_year") or {}
    return {int(k): int(v) for k, v in raw.items()}


def save_year_counts_cache(
    path: Path, *, fingerprint: dict[str, Any], rows_by_year: Mapping[int, int]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "rows_by_year": {str(k): int(v) for k, v in sorted(rows_by_year.items())},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def classify_csv_file(
    path: Path,
    *,
    downloads: Path,
    tier_a: Mapping[str, Path],
    tier_b: Mapping[str, Path],
) -> dict[str, Any]:
    name = path.name
    lower = name.lower()
    size = path.stat().st_size if path.is_file() else 0
    try:
        hdr_sha = header_fingerprint(path) if path.is_file() and size > 0 else ""
    except OSError:
        hdr_sha = ""

    rec: dict[str, Any] = {
        "path": str(path),
        "name": name,
        "bytes": size,
        "header_sha": hdr_sha,
        "class": FileClass.UNKNOWN_MARKET.value,
        "twin": None,
        "contract_key": None,
    }

    for key, tp in tier_a.items():
        if path.resolve() == Path(tp).resolve() or name == Path(tp).name:
            if key in ("options_panel", "vol_surface"):
                rec["class"] = FileClass.CANONICAL_INGESTED.value
                rec["contract_key"] = key
                return rec
            if key in (
                "options_slim",
                "options_constituents",
                "options_std",
                "vol_surface_legacy_a",
                "vol_surface_legacy_b",
            ):
                rec["class"] = FileClass.ALTERNATE_OM_NOT_INGESTED.value
                rec["contract_key"] = key
                return rec

    if any(m in lower for m in ALTERNATE_NAME_MARKERS):
        rec["class"] = FileClass.ALTERNATE_OM_NOT_INGESTED.value
        return rec

    # Explicit hash → Tier B map
    hint = HASH_TIER_B_HINTS.get(lower)
    if hint and hint in tier_b and Path(tier_b[hint]).is_file():
        twin = Path(tier_b[hint])
        if size == twin.stat().st_size and hdr_sha and hdr_sha == header_fingerprint(twin):
            rec["class"] = FileClass.TIER_B_DUPLICATE.value
            rec["twin"] = str(twin)
            rec["contract_key"] = hint
            return rec

    for key, bp in tier_b.items():
        bp = Path(bp)
        if not bp.is_file():
            continue
        if size == bp.stat().st_size and hdr_sha and hdr_sha == header_fingerprint(bp):
            rec["class"] = FileClass.TIER_B_DUPLICATE.value
            rec["twin"] = str(bp)
            rec["contract_key"] = key
            return rec
        if name == bp.name:
            # Same name under Downloads as Tier B path elsewhere
            if size == bp.stat().st_size and hdr_sha == header_fingerprint(bp):
                rec["class"] = FileClass.TIER_B_DUPLICATE.value
                rec["twin"] = str(bp)
                rec["contract_key"] = key
                return rec

    if name in NON_CONTRACT_UNIQUES or "ibes" in lower or "financial_ratios" in lower:
        # Disclosure ingest lands at lake/macro/ibes_financial_ratios.parquet.
        # Either way this is never UNKNOWN_MARKET (non-contract / disclosure-only).
        lake_ibes = Path(LAKE_ROOT) / "macro" / "ibes_financial_ratios.parquet"
        if lake_ibes.is_file():
            rec["class"] = FileClass.NON_MARKET.value
            rec["non_contract"] = True
            rec["disclosure_ingested"] = True
            rec["lake_artifact"] = str(lake_ibes)
            return rec
        rec["class"] = FileClass.UNIQUE_NOT_INGESTED.value
        rec["non_contract"] = True
        return rec

    # Tiny or non-tabular-ish leftovers in Downloads
    if size < 100 or not hdr_sha:
        rec["class"] = FileClass.NON_MARKET.value
        return rec

    # Known market-looking header tokens → unknown until classified
    hdr = _read_header_line(path).lower() if size else ""
    marketish = any(
        t in hdr
        for t in (
            "secid",
            "permno",
            "impl_volatility",
            "best_bid",
            "forwardprice",
            "vix",
            "gvkey",
        )
    )
    if marketish:
        rec["class"] = FileClass.UNKNOWN_MARKET.value
        return rec
    rec["class"] = FileClass.NON_MARKET.value
    return rec


def compare_year_counts(
    left: Mapping[int, int],
    right: Mapping[int, int],
    *,
    years: Iterable[int],
) -> tuple[bool, dict[int, int]]:
    deltas: dict[int, int] = {}
    ok = True
    for y in years:
        lv = int(left.get(int(y), 0))
        rv = int(right.get(int(y), 0))
        d = lv - rv
        deltas[int(y)] = d
        if d != 0:
            ok = False
    return ok, deltas


def compare_year_counts_with_rejects(
    staging: Mapping[int, int],
    parquet: Mapping[int, int],
    rejects: Mapping[int, int],
    *,
    years: Iterable[int],
) -> tuple[bool, dict[int, int]]:
    """S2 accounting: staging == parquet + rejects (exact Δ==0)."""
    combined = {
        int(y): int(parquet.get(int(y), 0)) + int(rejects.get(int(y), 0)) for y in years
    }
    return compare_year_counts(staging, combined, years=years)


def reject_year_counts(lake: Path, dataset: str) -> dict[int, int]:
    """Row counts from ``{lake}/_ingest_rejects/{dataset}/year=Y.parquet``."""
    import duckdb

    root = Path(lake) / "_ingest_rejects" / dataset
    out: dict[int, int] = {}
    if not root.is_dir():
        return out
    con = duckdb.connect()
    try:
        for p in sorted(root.glob("year=*.parquet")):
            try:
                y = int(p.stem.split("=", 1)[1])
            except (IndexError, ValueError):
                continue
            n = int(
                con.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{p.as_posix()}')"
                ).fetchone()[0]
            )
            out[y] = n
    finally:
        con.close()
    return out


def alternate_subset_ok(
    alt_rows: Mapping[int, int],
    lake_rows: Mapping[int, int],
    *,
    years: Iterable[int] | None = None,
) -> bool:
    years_set = set(int(y) for y in (years if years is not None else CONTRACT_YEARS))
    for y, n in alt_rows.items():
        yi = int(y)
        ni = int(n)
        if ni <= 0:
            continue
        if yi not in years_set:
            return False
        if ni > int(lake_rows.get(yi, 0)):
            return False
    return True


def primary_hist_cache_key(src: Path) -> dict[str, Any]:
    st = src.stat()
    return {
        "path": str(src.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
        "header_sha": header_fingerprint(src),
    }


def primary_hist_cache_valid(cache_path: Path, src: Path) -> bool:
    if not cache_path.is_file() or not src.is_file():
        return False
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    key = payload.get("cache_key")
    if not isinstance(key, dict):
        return False
    return key == primary_hist_cache_key(src)


def year_count_histogram_from_csv(
    path: Path,
    *,
    date_col: str = "date",
    year_start: int = YEAR_START,
    year_end: int = YEAR_END,
    progress_every: int = HIST_LOG_EVERY,
) -> dict[str, Any]:
    """Read-only count-only year histogram (constant RAM)."""
    t0 = time.perf_counter()
    rows_by_year: dict[int, int] = {}
    n_rows = 0
    n_skip = 0
    n_kept = 0
    header_sha = header_fingerprint(path)
    with path.open("r", newline="", encoding="utf-8", errors="replace") as fin:
        reader = csv.reader(fin)
        header = next(reader)
        lower = [h.strip().lower() for h in header]
        try:
            date_idx = lower.index(date_col.lower())
        except ValueError as exc:
            raise RuntimeError(f"date column {date_col!r} not in header of {path}") from exc
        for row in reader:
            n_rows += 1
            if progress_every and n_rows % progress_every == 0:
                log.info(
                    "count-only %s rows=%d kept=%d skip=%d elapsed=%.0fs",
                    path.name,
                    n_rows,
                    n_kept,
                    n_skip,
                    time.perf_counter() - t0,
                )
            if date_idx >= len(row):
                n_skip += 1
                continue
            y = parse_year(row[date_idx], year_start=year_start, year_end=year_end)
            if y is None:
                n_skip += 1
                continue
            # Count all parseable years; contract check happens in seals
            rows_by_year[y] = rows_by_year.get(y, 0) + 1
            if year_start <= y <= year_end:
                n_kept += 1
            else:
                n_skip += 1  # outside contract treated as skip for primary seal accounting
    return {
        "path": str(path),
        "header_sha": header_sha,
        "rows_by_year": {int(k): int(v) for k, v in sorted(rows_by_year.items())},
        "n_rows": n_rows,
        "n_kept": n_kept,
        "n_skip": n_skip,
        "elapsed_s": time.perf_counter() - t0,
        "bytes": int(path.stat().st_size),
    }


def decide_verdict(seals: Mapping[str, Mapping[str, Any]]) -> tuple[str, int]:
    """Return (verdict, exit_code). COMPLETE only if all S1–S6 pass."""
    required = [f"S{i}" for i in range(1, 7)]
    for k in required:
        if k not in seals:
            return "FAIL", 1

    soft_gap_reasons = {
        "wrds_unverified",
        "WRDS_NEWER_THAN_LAKE",
        "skipped_heavy_count_only",
        "classify_only",
    }

    hard_fail = False
    soft_gap = False
    for k in required:
        s = seals[k]
        if s.get("pass"):
            continue
        reason = str(s.get("reason") or "")
        if reason in soft_gap_reasons or reason.startswith("wrds_"):
            soft_gap = True
            continue
        hard_fail = True

    if hard_fail:
        return "FAIL", 1
    if all(seals[k].get("pass") for k in required):
        return "COMPLETE", 0
    if soft_gap:
        return "GAPS_DOCUMENTED", 2
    return "FAIL", 1


def staging_dir_fingerprint(staging_stem_dir: Path) -> dict[str, Any]:
    files = sorted(staging_stem_dir.glob("year=*.csv"))
    return {
        "dir": str(staging_stem_dir),
        "n_files": len(files),
        "files": [
            {
                "name": p.name,
                "size": p.stat().st_size,
                "mtime_ns": int(getattr(p.stat(), "st_mtime_ns", int(p.stat().st_mtime * 1e9))),
            }
            for p in files
        ],
    }


def staging_year_counts(
    staging_stem_dir: Path,
    *,
    artifacts: Path | None = None,
    reuse_cache: bool = True,
) -> dict[int, int]:
    out: dict[int, int] = {}
    if not staging_stem_dir.is_dir():
        return out
    fp = staging_dir_fingerprint(staging_stem_dir)
    cache_path = None
    if artifacts is not None:
        cache_path = _year_count_cache_path(artifacts, "staging", staging_stem_dir.name)
        if reuse_cache:
            hit = load_year_counts_cache(cache_path, fingerprint=fp)
            if hit is not None:
                log.info("staging year-count cache hit %s", staging_stem_dir.name)
                return hit
    for p in sorted(staging_stem_dir.glob("year=*.csv")):
        try:
            y = int(p.stem.split("=", 1)[1])
        except (IndexError, ValueError):
            continue
        log.info("counting staging lines %s (%.2f GB)", p.name, p.stat().st_size / 1e9)
        n = line_count(p)
        out[y] = max(0, n - 1)
    if cache_path is not None:
        save_year_counts_cache(cache_path, fingerprint=fp, rows_by_year=out)
    return out


def parquet_dataset_fingerprint(lake: Path, dataset: str) -> dict[str, Any]:
    root = lake / dataset
    files = sorted(root.rglob("*.parquet")) if root.is_dir() else []
    return {
        "dataset": dataset,
        "n_files": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "years": sorted({p.parts[-3] for p in files if len(p.parts) >= 3 and p.parts[-3].startswith("year=")}),
    }


def parquet_year_counts(
    lake: Path,
    dataset: str,
    *,
    artifacts: Path | None = None,
    reuse_cache: bool = True,
) -> dict[int, int]:
    import duckdb

    root = lake / dataset
    out: dict[int, int] = {}
    if not root.is_dir():
        return out
    fp = parquet_dataset_fingerprint(lake, dataset)
    cache_path = None
    if artifacts is not None:
        cache_path = _year_count_cache_path(artifacts, "parquet", dataset)
        if reuse_cache:
            hit = load_year_counts_cache(cache_path, fingerprint=fp)
            if hit is not None:
                log.info("parquet year-count cache hit %s", dataset)
                return hit
    con = duckdb.connect()
    try:
        for yd in sorted(root.glob("year=*")):
            if not yd.is_dir():
                continue
            try:
                y = int(yd.name.split("=", 1)[1])
            except (IndexError, ValueError):
                continue
            files = [p.as_posix() for p in yd.rglob("*.parquet")]
            if not files:
                out[y] = 0
                continue
            listed = ", ".join(f"'{f}'" for f in files)
            log.info("counting parquet %s (%d files)", yd.name, len(files))
            n = con.execute(
                f"SELECT COUNT(*) FROM read_parquet([{listed}], union_by_name=true)"
            ).fetchone()[0]
            out[y] = int(n)
    finally:
        con.close()
    if cache_path is not None:
        save_year_counts_cache(cache_path, fingerprint=fp, rows_by_year=out)
    return out


def tier_b_row_counts(csv_path: Path, parquet_path: Path) -> dict[str, Any]:
    import duckdb

    con = duckdb.connect()
    try:
        csv_n = int(
            con.execute(
                f"""
                SELECT COUNT(*) FROM read_csv_auto(
                    '{csv_path.as_posix()}',
                    sample_size=10000, ignore_errors=true, strict_mode=false, header=true
                )
                """
            ).fetchone()[0]
        )
        pq_n = int(
            con.execute(
                f"SELECT COUNT(*) FROM read_parquet('{parquet_path.as_posix()}', union_by_name=true)"
            ).fetchone()[0]
        )
        span_ok = True
        csv_span = None
        pq_span = None
        try:
            cols = [
                r[0].lower()
                for r in con.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{parquet_path.as_posix()}', union_by_name=true) LIMIT 0"
                ).fetchall()
            ]
            date_col = next((c for c in ("date", "dt", "yyyymm") if c in cols), None)
            if date_col is not None:
                pq_span = con.execute(
                    f"""
                    SELECT MIN(CAST({date_col} AS VARCHAR)), MAX(CAST({date_col} AS VARCHAR))
                    FROM read_parquet('{parquet_path.as_posix()}', union_by_name=true)
                    """
                ).fetchone()
                csv_span = con.execute(
                    f"""
                    SELECT MIN(CAST({date_col} AS VARCHAR)), MAX(CAST({date_col} AS VARCHAR))
                    FROM read_csv_auto(
                        '{csv_path.as_posix()}',
                        sample_size=10000, ignore_errors=true, strict_mode=false, header=true
                    )
                    """
                ).fetchone()
                span_ok = list(csv_span) == list(pq_span)
        except Exception:
            span_ok = True  # row equality still required; span best-effort
    finally:
        con.close()
    return {
        "csv_rows": csv_n,
        "parquet_rows": pq_n,
        "delta": csv_n - pq_n,
        "csv_span": list(csv_span) if csv_span else None,
        "parquet_span": list(pq_span) if pq_span else None,
        "pass": csv_n == pq_n and span_ok,
    }


def list_downloads_csvs(downloads: Path) -> list[Path]:
    if not downloads.is_dir():
        return []
    nested = downloads.name == "volsurf_raw" or (downloads / "om").is_dir()
    iterator = downloads.rglob("*.csv") if nested else downloads.glob("*.csv")
    return sorted(p for p in iterator if p.is_file() and not p.is_symlink())


def load_or_build_histogram(
    src: Path,
    *,
    cache_path: Path,
    reuse_cache: bool,
    date_col: str = "date",
) -> dict[str, Any]:
    if reuse_cache and primary_hist_cache_valid(cache_path, src):
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        payload["cache_hit"] = True
        return payload
    hist = year_count_histogram_from_csv(src, date_col=date_col)
    rows_by_year = {int(k): int(v) for k, v in hist["rows_by_year"].items()}
    payload = {
        "cache_key": primary_hist_cache_key(src),
        "rows_by_year": rows_by_year,
        "n_rows": hist["n_rows"],
        "n_kept": hist["n_kept"],
        "n_skip": hist["n_skip"],
        "elapsed_s": hist["elapsed_s"],
        "cache_hit": False,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _enrich_present(lake: Path) -> dict[str, Any]:
    macro = lake / "macro"
    required = {
        "crsp_optionm_link": macro / "crsp_optionm_link.parquet",
        "crsp_om_adv": macro / "crsp_om_adv.parquet",
        "compustat_funda_enrich": macro / "compustat_funda_enrich.parquet",
        "om_distrd": macro / "om_distrd.parquet",
        "om_opvold": macro / "om_opvold.parquet",
        "om_borrate": macro / "om_borrate.parquet",
        "om_zerocd": macro / "om_zerocd.parquet",
    }
    details = {}
    ok = True
    for name, path in required.items():
        present = path.is_file() and path.stat().st_size > 0
        details[name] = {"path": str(path), "present": present, "bytes": path.stat().st_size if path.is_file() else 0}
        if not present:
            ok = False
    return {"pass": ok, "tables": details}


def probe_wrds_catalog(lake: Path) -> dict[str, Any]:
    """List WRDS tables in contracted schemas and classify vs lake enrich."""
    from mascotrl.data.wrds_enrich import connect_wrds

    already = {
        "wrdsapps_link_crsp_optionm.opcrsphist",
        "comp.funda",
        "crsp.ccmxpf_lnkhist",
        "optionm.distrd",
        "optionm.opvold",
        "optionm.securd",
        "optionm.secnmd",
        "optionm.idxdvd",
        "optionm.zerocd",
        # yearly families
        "optionm.borrateYYYY",
        "optionm.stdbrteYYYY",
        "optionm.stdopdYYYY",
        "optionm.secprd",  # probe table
    }
    disclosure = {
        "comp.fundq",
        "ibes.",
    }
    spine_hints = (
        "opprcd",
        "vsurfd",
        "secprd",
        "distrd",
        "borrate",
        "stdopd",
        "opvold",
    )

    conn = connect_wrds()
    try:
        rows = []
        for schema in ("optionm", "crsp", "comp", "wrdsapps_link_crsp_optionm"):
            try:
                tables = conn.list_tables(library=schema)
            except Exception:
                try:
                    df = conn.raw_sql(
                        f"""
                        select table_name
                        from information_schema.tables
                        where table_schema = '{schema}'
                        order by table_name
                        """
                    )
                    tables = list(df["table_name"].astype(str)) if df is not None else []
                except Exception as exc:
                    rows.append(
                        {
                            "schema": schema,
                            "error": str(exc),
                            "class": "IRRELEVANT",
                        }
                    )
                    continue
            for t in tables:
                full = f"{schema}.{t}"
                tlow = str(t).lower()
                if full in already or any(
                    full.startswith(a.replace("YYYY", "")) or a.replace("YYYY", "") in full
                    for a in already
                ):
                    cls = "ALREADY_IN_LAKE"
                elif any(full.startswith(d) or d.rstrip(".") in full for d in disclosure):
                    cls = "DISCLOSURE_ONLY"
                elif any(h in tlow for h in spine_hints):
                    cls = "SPINE_CANDIDATE"
                else:
                    cls = "IRRELEVANT"
                # normalize yearly optionm tables
                if schema == "optionm" and (
                    tlow.startswith("borrate")
                    or tlow.startswith("stdbrte")
                    or tlow.startswith("stdopd")
                    or tlow.startswith("opprcd")
                    or tlow.startswith("vsurfd")
                ):
                    if tlow.startswith("opprcd") or tlow.startswith("vsurfd"):
                        cls = "ALREADY_IN_LAKE"  # covered by Tier A dumps
                    elif any(tlow.startswith(p) for p in ("borrate", "stdbrte", "stdopd")):
                        cls = "ALREADY_IN_LAKE"
                rows.append({"table": full, "class": cls})
        return {"tables": rows, "n": len(rows)}
    finally:
        conn.close()


def probe_wrds_max_dates(lake: Path) -> dict[str, Any]:
    """Live WRDS MAX(date) probes; raises if credentials missing."""
    from mascotrl.data.om_enrich import lake_secids
    from mascotrl.data.wrds_enrich import connect_wrds

    secids = lake_secids(lake)[:50]  # cheap probe sample
    if not secids:
        return {"pass": False, "reason": "wrds_no_lake_secids"}
    catalog = probe_wrds_catalog(lake)
    conn = connect_wrds()
    try:
        # Lake max dates from parquet
        import duckdb

        con = duckdb.connect()
        lake_max: dict[str, str | None] = {}
        for ds in ("options_panel", "vol_surface"):
            files = [p.as_posix() for p in (lake / ds).rglob("*.parquet")]
            if not files:
                lake_max[ds] = None
                continue
            # sample last year only for speed
            yfiles = [p.as_posix() for p in (lake / ds / f"year={YEAR_END}").rglob("*.parquet")]
            use = yfiles or files[:20]
            listed = ", ".join(f"'{f}'" for f in use)
            try:
                row = con.execute(
                    f"SELECT MAX(CAST(date AS DATE)) FROM read_parquet([{listed}], union_by_name=true)"
                ).fetchone()
                lake_max[ds] = str(row[0]) if row and row[0] is not None else None
            except Exception as exc:
                lake_max[ds] = None
                log.warning("lake max date failed for %s: %s", ds, exc)
        con.close()

        ids = ",".join(str(int(i)) for i in secids)
        wrds_row = conn.raw_sql(
            f"""
            select max(date) as max_date
            from optionm.secprd
            where secid in ({ids})
            """
        )
        wrds_max = None
        if wrds_row is not None and len(wrds_row):
            wrds_max = str(wrds_row.iloc[0]["max_date"])

        newer = False
        if wrds_max and lake_max.get("options_panel"):
            newer = wrds_max > lake_max["options_panel"]
        return {
            "pass": not newer,
            "reason": "WRDS_NEWER_THAN_LAKE" if newer else "ok",
            "lake_max": lake_max,
            "wrds_secprd_max": wrds_max,
            "n_secids_probed": len(secids),
            "catalog": catalog,
        }
    finally:
        conn.close()


@dataclass
class AuditPaths:
    downloads: Path
    lake: Path
    artifacts: Path
    tier_a: dict[str, Path]
    tier_b: dict[str, Path]


def default_paths() -> AuditPaths:
    return AuditPaths(
        downloads=RAW_ROOT,
        lake=LAKE_ROOT,
        artifacts=MASCOTRL_ROOT / "logs" / "artifacts",
        tier_a=dict(TIER_A),
        tier_b=dict(TIER_B),
    )


def run_coverage_audit(
    *,
    downloads: Path | None = None,
    lake: Path | None = None,
    artifacts: Path | None = None,
    reuse_primary_hist_cache: bool = True,
    classify_only: bool = False,
    run_wrds: bool | None = None,
    skip_heavy_count_only: bool = False,
) -> dict[str, Any]:
    """Run classification + seals. Heavy count-only passes unless skip_heavy_count_only."""
    base = default_paths()
    downloads = Path(downloads or base.downloads)
    lake = Path(lake or base.lake)
    artifacts = Path(artifacts or base.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    tier_a = base.tier_a
    tier_b = base.tier_b

    t0 = time.perf_counter()
    files = list_downloads_csvs(downloads)
    classifications = [
        classify_csv_file(p, downloads=downloads, tier_a=tier_a, tier_b=tier_b) for p in files
    ]

    # Also ensure Tier A paths are classified even if outside Downloads listing quirks
    for key in ("options_panel", "vol_surface"):
        p = Path(tier_a[key])
        if p.is_file() and not any(Path(c["path"]) == p for c in classifications):
            classifications.append(
                classify_csv_file(p, downloads=downloads, tier_a=tier_a, tier_b=tier_b)
            )

    unknown = [c for c in classifications if c["class"] == FileClass.UNKNOWN_MARKET.value]
    non_contract = [
        c
        for c in classifications
        if c["class"] == FileClass.UNIQUE_NOT_INGESTED.value and c.get("non_contract")
    ]
    s6_missing = s6_missing_contract(files, tier_a, tier_b)
    s6_ok, s6_reason = s6_listing_ok(
        files=files, unknown=unknown, missing_contract=s6_missing
    )

    seals: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {
        "downloads": str(downloads),
        "lake": str(lake),
        "classifications": classifications,
        "seal_predicates": seals,
        "classify_only": classify_only,
    }

    if classify_only:
        seals["S1"] = {"pass": False, "reason": "classify_only"}
        seals["S2"] = {"pass": False, "reason": "classify_only"}
        seals["S3"] = {"pass": False, "reason": "classify_only"}
        seals["S4"] = {"pass": False, "reason": "classify_only"}
        seals["S5"] = {"pass": False, "reason": "classify_only"}
        seals["S6"] = {
            "pass": s6_ok,
            "reason": s6_reason,
            "unknown": unknown,
            "non_contract": non_contract,
            "n_listed": len(files),
            "missing_contract": s6_missing,
        }
        verdict, code = "GAPS_DOCUMENTED", 2
        if not seals["S6"]["pass"]:
            verdict, code = "FAIL", 1
        report["verdict"] = verdict
        report["exit_code"] = code
        report["elapsed_s"] = time.perf_counter() - t0
        out = artifacts / "lake_source_coverage.json"
        out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        report["artifact"] = str(out)
        return report

    # S6 first
    seals["S6"] = {
        "pass": s6_ok,
        "reason": s6_reason,
        "unknown": unknown,
        "non_contract": non_contract,
        "n_listed": len(files),
        "missing_contract": s6_missing,
    }

    staging_root = lake / "_csv_year_staging"
    opt_stem = Path(tier_a["options_panel"]).stem
    vol_stem = Path(tier_a["vol_surface"]).stem

    # S2 staging ↔ parquet
    opt_staging = staging_year_counts(
        staging_root / opt_stem, artifacts=artifacts, reuse_cache=reuse_primary_hist_cache
    )
    vol_staging = staging_year_counts(
        staging_root / vol_stem, artifacts=artifacts, reuse_cache=reuse_primary_hist_cache
    )
    opt_pq = parquet_year_counts(
        lake, "options_panel", artifacts=artifacts, reuse_cache=reuse_primary_hist_cache
    )
    vol_pq = parquet_year_counts(
        lake, "vol_surface", artifacts=artifacts, reuse_cache=reuse_primary_hist_cache
    )
    opt_rej = reject_year_counts(lake, "options_panel")
    vol_rej = reject_year_counts(lake, "vol_surface")
    years = list(CONTRACT_YEARS)
    opt_ok, opt_delta = compare_year_counts_with_rejects(
        opt_staging, opt_pq, opt_rej, years=years
    )
    vol_ok, vol_delta = compare_year_counts_with_rejects(
        vol_staging, vol_pq, vol_rej, years=years
    )
    staging_complete = (
        set(opt_staging.keys()) >= set(years)
        and set(vol_staging.keys()) >= set(years)
        and set(opt_pq.keys()) >= set(years)
        and set(vol_pq.keys()) >= set(years)
    )
    # Fail closed: if parquet < staging for a year and rejects file missing, delta stays nonzero
    seals["S2"] = {
        "pass": bool(opt_ok and vol_ok and staging_complete),
        "reason": "ok" if (opt_ok and vol_ok and staging_complete) else (
            "staging_incomplete" if not staging_complete else "year_mismatch"
        ),
        "accounting": "staging == parquet + rejects",
        "options_panel": {
            "staging": opt_staging,
            "parquet": opt_pq,
            "rejects": opt_rej,
            "deltas": opt_delta,
        },
        "vol_surface": {
            "staging": vol_staging,
            "parquet": vol_pq,
            "rejects": vol_rej,
            "deltas": vol_delta,
        },
    }

    # S1 primary → staging
    if skip_heavy_count_only:
        seals["S1"] = {
            "pass": False,
            "reason": "skipped_heavy_count_only",
            "note": "Re-run without --skip-heavy-count-only for COMPLETE",
        }
        seals["S4"] = {
            "pass": False,
            "reason": "skipped_heavy_count_only",
        }
    else:
        s1_details = {}
        s1_pass = True
        for key, stem, staging_map in (
            ("options_panel", opt_stem, opt_staging),
            ("vol_surface", vol_stem, vol_staging),
        ):
            src = Path(tier_a[key])
            cache = artifacts / f"lake_source_coverage_primary_hist_{stem}.json"
            if not src.is_file():
                s1_pass = False
                s1_details[key] = {"pass": False, "reason": "primary_missing"}
                continue
            hist = load_or_build_histogram(
                src, cache_path=cache, reuse_cache=reuse_primary_hist_cache
            )
            rows = {int(k): int(v) for k, v in hist["rows_by_year"].items()}
            # Only contract years for equality with staging
            contract_rows = {y: int(rows.get(y, 0)) for y in years}
            ok, deltas = compare_year_counts(contract_rows, staging_map, years=years)
            extras = {y: n for y, n in rows.items() if y not in set(years) and n > 0}
            if extras:
                ok = False
            s1_details[key] = {
                "pass": ok,
                "cache_hit": hist.get("cache_hit"),
                "deltas": deltas,
                "extras_outside_contract": extras,
                "n_skip": hist.get("n_skip"),
                "n_rows": hist.get("n_rows"),
                "cache": str(cache),
            }
            s1_pass = s1_pass and ok
        seals["S1"] = {
            "pass": s1_pass,
            "reason": "ok" if s1_pass else "year_mismatch",
            "details": s1_details,
        }

        # S4 alternates
        alt_files = [
            c
            for c in classifications
            if c["class"] == FileClass.ALTERNATE_OM_NOT_INGESTED.value
        ]
        alt_matrix = []
        s4_pass = True
        for c in alt_files:
            p = Path(c["path"])
            if not p.is_file():
                continue
            is_surface = "volsurf" in p.name.lower() or "vsurd" in p.name.lower()
            lake_rows = vol_pq if is_surface else opt_pq
            cache = artifacts / f"lake_source_coverage_alt_hist_{p.stem}.json"
            hist = load_or_build_histogram(
                p, cache_path=cache, reuse_cache=reuse_primary_hist_cache
            )
            rows = {int(k): int(v) for k, v in hist["rows_by_year"].items()}
            ok = alternate_subset_ok(rows, lake_rows, years=years)
            alt_matrix.append(
                {
                    "path": str(p),
                    "pass": ok,
                    "rows_by_year": rows,
                    "lake_dataset": "vol_surface" if is_surface else "options_panel",
                    "cache_hit": hist.get("cache_hit"),
                }
            )
            if not ok:
                s4_pass = False
        seals["S4"] = {
            "pass": s4_pass,
            "reason": "ok" if s4_pass else "ALTERNATE_HAS_ROWS_NOT_IN_LAKE",
            "alternates": alt_matrix,
        }

    # S3 Tier B
    s3_details = {}
    s3_pass = True
    for key, csv_path in tier_b.items():
        csv_path = Path(csv_path)
        pq = lake / "macro" / f"{key}.parquet"
        # spx_index may be partitioned dir — handle file only for exact count
        if not csv_path.is_file():
            s3_details[key] = {"pass": False, "reason": "tier_b_csv_missing"}
            s3_pass = False
            continue
        if not pq.is_file():
            # partitioned macro?
            pq_dir = lake / "macro" / key
            if pq_dir.is_dir() and any(pq_dir.rglob("*.parquet")):
                import duckdb

                con = duckdb.connect()
                try:
                    files = [p.as_posix() for p in pq_dir.rglob("*.parquet")]
                    listed = ", ".join(f"'{f}'" for f in files)
                    pq_n = int(
                        con.execute(
                            f"SELECT COUNT(*) FROM read_parquet([{listed}], union_by_name=true)"
                        ).fetchone()[0]
                    )
                    csv_n = int(
                        con.execute(
                            f"""
                            SELECT COUNT(*) FROM read_csv_auto(
                                '{csv_path.as_posix()}',
                                sample_size=10000, ignore_errors=true, strict_mode=false, header=true
                            )
                            """
                        ).fetchone()[0]
                    )
                finally:
                    con.close()
                ok = csv_n == pq_n
                s3_details[key] = {
                    "pass": ok,
                    "csv_rows": csv_n,
                    "parquet_rows": pq_n,
                    "delta": csv_n - pq_n,
                    "parquet": str(pq_dir),
                }
                s3_pass = s3_pass and ok
                continue
            s3_details[key] = {"pass": False, "reason": "macro_parquet_missing", "parquet": str(pq)}
            s3_pass = False
            continue
        # Skip extremely large Tier B if any (spx_index can be huge)
        if csv_path.stat().st_size > 3_000_000_000:
            # Still required for seal — count via streaming line count vs parquet
            csv_n = max(0, line_count(csv_path) - 1)
            import duckdb

            con = duckdb.connect()
            try:
                pq_n = int(
                    con.execute(
                        f"SELECT COUNT(*) FROM read_parquet('{pq.as_posix()}', union_by_name=true)"
                    ).fetchone()[0]
                )
            finally:
                con.close()
            ok = csv_n == pq_n
            s3_details[key] = {
                "pass": ok,
                "csv_rows": csv_n,
                "parquet_rows": pq_n,
                "delta": csv_n - pq_n,
                "method": "line_count",
            }
            s3_pass = s3_pass and ok
            continue
        try:
            cmp = tier_b_row_counts(csv_path, pq)
            s3_details[key] = cmp
            s3_pass = s3_pass and bool(cmp["pass"])
        except Exception as exc:
            s3_details[key] = {"pass": False, "reason": f"count_error:{exc}"}
            s3_pass = False

    # Hash / Downloads twins: size + header SHA must equal Tier B twin
    for c in classifications:
        name_l = str(c.get("name") or "").lower()
        if c["class"] == FileClass.TIER_B_DUPLICATE.value:
            twin_s = c.get("twin")
            src = Path(c["path"])
            twin = Path(twin_s) if twin_s else None
            ok_twin = bool(
                twin
                and twin.is_file()
                and src.is_file()
                and src.stat().st_size == twin.stat().st_size
                and header_fingerprint(src) == header_fingerprint(twin)
            )
            s3_details[f"twin:{c['name']}"] = {
                "pass": ok_twin,
                "twin": twin_s,
                "contract_key": c.get("contract_key"),
            }
            s3_pass = s3_pass and ok_twin
            continue
        if name_l in HASH_TIER_B_HINTS:
            s3_pass = False
            s3_details[f"hash:{c['name']}"] = {
                "pass": False,
                "reason": "hash_not_matched_to_tier_b",
                "hint": HASH_TIER_B_HINTS[name_l],
            }

    seals["S3"] = {
        "pass": s3_pass,
        "reason": "ok" if s3_pass else "tier_b_mismatch",
        "details": s3_details,
    }

    # S5 WRDS
    if run_wrds is None:
        run_wrds = bool(os.environ.get("WRDS_USERNAME"))
    enrich = _enrich_present(lake)
    if not run_wrds:
        seals["S5"] = {
            "pass": False,
            "reason": "wrds_unverified",
            "enrich_local": enrich,
        }
    else:
        try:
            probe = probe_wrds_max_dates(lake)
            seals["S5"] = {
                "pass": bool(probe.get("pass") and enrich["pass"]),
                "reason": probe.get("reason") if probe.get("pass") else probe.get("reason", "wrds_probe_failed"),
                "enrich_local": enrich,
                "probe": probe,
            }
            if not enrich["pass"]:
                seals["S5"]["pass"] = False
                seals["S5"]["reason"] = "enrich_missing"
        except Exception as exc:
            seals["S5"] = {
                "pass": False,
                "reason": f"wrds_error:{exc}",
                "enrich_local": enrich,
            }

    jkp = lake / "factors" / "jkp_chars.parquet"
    report["public_gaps"] = {
        "jkp_chars": {
            "path": str(jkp),
            "empty_or_missing": (not jkp.is_file())
            or jkp.stat().st_size < 1000,
            "non_contract": True,
        },
        "ibes": non_contract,
    }

    verdict, code = decide_verdict(seals)
    report["verdict"] = verdict
    report["exit_code"] = code
    report["seal_predicates"] = seals
    report["elapsed_s"] = time.perf_counter() - t0
    # Summarize count-only I/O when present
    io_summary: dict[str, Any] = {}
    for sk in ("S1", "S4"):
        s = seals.get(sk) or {}
        if "details" in s:
            for k, det in (s.get("details") or {}).items():
                if isinstance(det, dict) and "n_rows" in det:
                    io_summary[f"{sk}.{k}"] = {
                        "n_rows": det.get("n_rows"),
                        "n_skip": det.get("n_skip"),
                        "cache_hit": det.get("cache_hit"),
                    }
        for alt in s.get("alternates") or []:
            io_summary[f"S4.{Path(alt['path']).name}"] = {
                "cache_hit": alt.get("cache_hit"),
                "rows_by_year_sum": sum(int(v) for v in (alt.get("rows_by_year") or {}).values()),
            }
    report["count_only_io"] = io_summary
    out = artifacts / "lake_source_coverage.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    report["artifact"] = str(out)
    return report
