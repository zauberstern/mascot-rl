#!/usr/bin/env python3
"""Materialize full Compustat funda/fundq/company/CCM under macro/wrds/."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mascotrl.data.paths import LAKE_ROOT, MASCOTRL_ROOT  # noqa: E402
from mascotrl.data.wrds_enrich import (  # noqa: E402
    _load_dotenv_files,
    materialize_wrds_fundamentals_full,
)
from mascotrl.logging_utils import setup_logging  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lake-dir", default=None)
    p.add_argument("--year-start", type=int, default=2003)
    p.add_argument("--year-end", type=int, default=2024)
    p.add_argument("--no-fundq", action="store_true")
    p.add_argument("--no-spine", action="store_true")
    p.add_argument("--log-file", default=str(MASCOTRL_ROOT / "logs" / "wrds_fundamentals.log"))
    args = p.parse_args()
    log = setup_logging(log_file=args.log_file)
    _load_dotenv_files()
    lake = Path(args.lake_dir) if args.lake_dir else LAKE_ROOT
    try:
        result = materialize_wrds_fundamentals_full(
            lake,
            years=range(args.year_start, args.year_end + 1),
            include_fundq=not args.no_fundq,
            regenerate_spine_enrich=not args.no_spine,
        )
    except Exception as exc:
        log.error("materialize failed: %s", exc)
        sys.exit(2)
    log.info("counts=%s", result.get("counts"))
    for k, path in (result.get("paths") or {}).items():
        log.info("  %s → %s", k, path)
    sys.exit(0)


if __name__ == "__main__":
    main()
