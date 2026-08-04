#!/usr/bin/env python3
"""Force reconvert lake years from staging (no monolith re-split).

Preserves hive wiring; writes rejects under ``_ingest_rejects/`` only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.lake_builder import ParquetDataLakeBuilder  # noqa: E402
from src.data.paths import LAKE_ROOT, MASCOTRL_ROOT  # noqa: E402
from src.logging_utils import setup_logging  # noqa: E402

MISMATCH_DEFAULT = {
    "options_panel": [
        2003,
        2004,
        2005,
        2006,
        2008,
        2009,
        2010,
        2011,
        2018,
        2019,
        2020,
        2021,
        2022,
        2023,
        2024,
    ],
    "vol_surface": [
        2003,
        2004,
        2005,
        2006,
        2008,
        2009,
        2010,
        2011,
        2018,
        2019,
        2020,
        2021,
        2022,
        2023,
        2024,
    ],
}


def years_from_seal(artifact: Path, dataset: str) -> list[int]:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    s2 = (payload.get("seal_predicates") or {}).get("S2") or {}
    block = s2.get(dataset) or {}
    deltas = block.get("deltas") or {}
    return sorted(int(y) for y, d in deltas.items() if int(d) != 0)


def bust_parquet_year_caches(artifacts: Path) -> None:
    for name in (
        "lake_source_coverage_parquet_year_counts_options_panel.json",
        "lake_source_coverage_parquet_year_counts_vol_surface.json",
    ):
        p = artifacts / name
        if p.is_file():
            p.unlink()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["options_panel", "vol_surface", "both"], default="both")
    p.add_argument("--years", default=None, help="Comma-separated years")
    p.add_argument(
        "--from-seal",
        action="store_true",
        help="Read mismatch years from seal artifact deltas",
    )
    p.add_argument(
        "--seal-artifact",
        default=str(MASCOTRL_ROOT / "logs" / "artifacts" / "lake_source_coverage.json"),
    )
    p.add_argument("--lake-dir", default=None)
    p.add_argument("--no-backup", action="store_true")
    p.add_argument("--verify", action="store_true", help="Fail if any year pass=False")
    p.add_argument("--log-file", default=str(MASCOTRL_ROOT / "logs" / "reconvert_lake_years.log"))
    args = p.parse_args()
    log = setup_logging(log_file=args.log_file)

    datasets = (
        ["options_panel", "vol_surface"] if args.dataset == "both" else [args.dataset]
    )
    builder = ParquetDataLakeBuilder(lake_base_dir=args.lake_dir)
    all_ok = True
    reports = []
    for ds in datasets:
        if args.years:
            years = [int(x.strip()) for x in args.years.split(",") if x.strip()]
        elif args.from_seal:
            seal = Path(args.seal_artifact)
            if seal.is_file():
                years = years_from_seal(seal, ds)
            else:
                years = list(MISMATCH_DEFAULT[ds])
            if not years:
                years = list(MISMATCH_DEFAULT[ds])
        else:
            years = list(MISMATCH_DEFAULT[ds])
        log.info("reconvert dataset=%s years=%s", ds, years)
        report = builder.reconvert_years(
            ds, years, force=True, backup=not args.no_backup
        )
        reports.append(report)
        ok = bool(report.get("pass"))
        all_ok = all_ok and ok
        log.info("reconvert dataset=%s pass=%s", ds, ok)

    artifacts = MASCOTRL_ROOT / "logs" / "artifacts"
    bust_parquet_year_caches(artifacts)
    out = artifacts / "lake_reconvert_years.json"
    out.write_text(json.dumps(reports, indent=2, default=str) + "\n", encoding="utf-8")
    log.info("wrote %s", out)

    if args.verify and not all_ok:
        sys.exit(1)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
