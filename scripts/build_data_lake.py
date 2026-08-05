#!/usr/bin/env python3
"""Build Parquet data lake from Tier A (mounted) + Tier B (local)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mascotrl.data.lake_builder import ParquetDataLakeBuilder
from mascotrl.data.paths import TIER_A, tier_a_available
from mascotrl.logging_utils import log_span, setup_logging

# Known S2 mismatch years from prior seal runs (scripts/reconvert_lake_years.py).
_MISMATCH_DEFAULT = {
    "options_panel": [
        2003, 2004, 2005, 2006, 2008, 2009, 2010, 2011,
        2018, 2019, 2020, 2021, 2022, 2023, 2024,
    ],
    "vol_surface": [
        2003, 2004, 2005, 2006, 2008, 2009, 2010, 2011,
        2018, 2019, 2020, 2021, 2022, 2023, 2024,
    ],
}


def _years_from_seal(seal_path: Path, dataset: str) -> list[int]:
    payload = json.loads(seal_path.read_text(encoding="utf-8"))
    s2 = (payload.get("seal_predicates") or {}).get("S2") or {}
    deltas = (s2.get(dataset) or {}).get("deltas") or {}
    return sorted(int(y) for y, d in deltas.items() if int(d) != 0)


def _reconvert_mismatch_years(builder: ParquetDataLakeBuilder, log) -> int:
    """Reconvert S2 mismatch years then re-audit (skip-heavy S1/S4)."""
    from mascotrl.data.lake_source_audit import run_coverage_audit

    artifacts = ROOT / "logs" / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    seal_path = artifacts / "lake_source_coverage.json"
    mismatched: dict[str, list[int]] = {}
    if seal_path.is_file():
        for ds in ("options_panel", "vol_surface"):
            years = _years_from_seal(seal_path, ds)
            mismatched[ds] = years
            log.info("S2 mismatch from seal dataset=%s years=%s", ds, years)
    else:
        mismatched = {k: list(v) for k, v in _MISMATCH_DEFAULT.items()}
        log.info("S2 mismatch from MISMATCH_DEFAULT (no seal yet)")

    reports: list[dict] = []
    all_ok = True
    for ds, years in mismatched.items():
        if not years:
            log.info("S2 already matched for %s", ds)
            continue
        report = builder.reconvert_years(ds, years, force=True, backup=True)
        reports.append(report)
        all_ok = all_ok and bool(report.get("pass"))
        log.info("reconvert dataset=%s pass=%s", ds, report.get("pass"))

    out = artifacts / "lake_reconvert_years.json"
    out.write_text(json.dumps(reports, indent=2, default=str) + "\n", encoding="utf-8")
    log.info("wrote %s", out)

    for name in (
        "lake_source_coverage_parquet_year_counts_options_panel.json",
        "lake_source_coverage_parquet_year_counts_vol_surface.json",
    ):
        cache = artifacts / name
        if cache.is_file():
            cache.unlink()

    result = run_coverage_audit(
        lake=builder.lake_base_dir,
        artifacts=artifacts,
        skip_heavy_count_only=True,
    )
    seals = (result or {}).get("seal_predicates") or {}
    s2 = seals.get("S2") or {}
    log.info("post-reconvert S2 pass=%s verdict=%s", s2.get("pass"), result.get("verdict"))
    if reports and not all_ok:
        return 1
    if s2 and s2.get("pass") is False:
        log.warning("S2 not green yet; see %s", artifacts / "lake_source_coverage.json")
    return 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lake-dir", default=None)
    p.add_argument("--tier-b-only", action="store_true")
    p.add_argument("--log-file", default=str(ROOT / "logs" / "data_lake.log"))
    p.add_argument(
        "--max-memory",
        default=None,
        help="DuckDB memory cap (default 8GB). Example: 8GB, 12GB",
    )
    p.add_argument("--threads", type=int, default=None, help="DuckDB threads (default 2)")
    p.add_argument(
        "--reconvert-mismatch-years",
        action="store_true",
        help="Force reconvert S2 mismatch years then re-audit (does not rebuild lake).",
    )
    args = p.parse_args()

    log = setup_logging(log_file=args.log_file)
    kwargs = {}
    if args.max_memory:
        kwargs["max_memory"] = args.max_memory
    if args.threads is not None:
        kwargs["threads"] = args.threads
    builder = ParquetDataLakeBuilder(lake_base_dir=args.lake_dir, **kwargs)

    if args.reconvert_mismatch_years:
        log.info("S2 reconvert path lake_dir=%s", builder.lake_base_dir)
        sys.exit(_reconvert_mismatch_years(builder, log))

    include_a = not args.tier_b_only
    log.info(
        "Lake build start lake_dir=%s include_tier_a=%s max_memory=%s threads=%s",
        builder.lake_base_dir,
        include_a,
        builder.max_memory,
        builder.threads,
    )
    if include_a:
        log.info("Tier A mount present=%s panel=%s", tier_a_available(), TIER_A["options_panel"])
        if not tier_a_available():
            log.error("Tier A not available; pass --tier-b-only or mount RAW_ROOT")
            sys.exit(1)
    with log_span(log, "L5.execute_full_build", include_tier_a=include_a):
        builder.execute_full_build(include_tier_a=include_a)
    log.info("Data lake build complete -> %s", builder.lake_base_dir)


if __name__ == "__main__":
    main()
