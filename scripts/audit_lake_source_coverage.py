#!/usr/bin/env python3
"""Fail-closed lake / WRDS source-coverage seal.

Writes logs/artifacts/lake_source_coverage.json
Exit 0 only for COMPLETE.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.lake_source_audit import run_coverage_audit  # noqa: E402
from src.data.wrds_enrich import _load_dotenv_files  # noqa: E402
from src.logging_utils import setup_logging  # noqa: E402


def main() -> None:
    _load_dotenv_files()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--downloads-dir", default=None)
    p.add_argument("--lake-dir", default=None)
    p.add_argument("--artifacts-dir", default=None)
    p.add_argument(
        "--classify-only",
        action="store_true",
        help="Inventory only (cannot emit COMPLETE)",
    )
    p.add_argument(
        "--skip-heavy-count-only",
        action="store_true",
        help="Skip primary/alternate count-only streams (cannot emit COMPLETE)",
    )
    p.add_argument(
        "--no-reuse-primary-hist-cache",
        action="store_true",
        help="Force re-stream primary/alternate histograms",
    )
    p.add_argument(
        "--wrds",
        action="store_true",
        help="Force WRDS probes (default: on when WRDS_USERNAME set)",
    )
    p.add_argument(
        "--no-wrds",
        action="store_true",
        help="Skip WRDS even if credentials exist",
    )
    p.add_argument("--log-file", default=str(ROOT / "logs" / "lake_source_audit.log"))
    args = p.parse_args()
    log = setup_logging(log_file=args.log_file)

    run_wrds: bool | None
    if args.no_wrds:
        run_wrds = False
    elif args.wrds:
        run_wrds = True
    else:
        run_wrds = None

    report = run_coverage_audit(
        downloads=Path(args.downloads_dir) if args.downloads_dir else None,
        lake=Path(args.lake_dir) if args.lake_dir else None,
        artifacts=Path(args.artifacts_dir) if args.artifacts_dir else None,
        reuse_primary_hist_cache=not args.no_reuse_primary_hist_cache,
        classify_only=args.classify_only,
        run_wrds=run_wrds,
        skip_heavy_count_only=args.skip_heavy_count_only,
    )
    log.info(
        "lake_source_coverage verdict=%s exit=%s artifact=%s",
        report.get("verdict"),
        report.get("exit_code"),
        report.get("artifact"),
    )
    for k, s in (report.get("seal_predicates") or {}).items():
        log.info("  %s pass=%s reason=%s", k, s.get("pass"), s.get("reason"))
    code = int(report.get("exit_code") or 1)
    sys.exit(code)


if __name__ == "__main__":
    main()
