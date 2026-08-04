#!/usr/bin/env python3
"""Audit lake column coverage: every column has a consumer or unused reason."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.feature_coverage import run_coverage_audit  # noqa: E402
from src.data.paths import LAKE_ROOT, assert_lake_mounted  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lake", default=str(LAKE_ROOT))
    p.add_argument(
        "--out",
        default=str(ROOT / "data" / "lake" / "_audit" / "column_coverage.json"),
    )
    args = p.parse_args()
    lake = assert_lake_mounted(Path(args.lake))
    report = run_coverage_audit(lake, out_path=Path(args.out))
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "n_tables": report["n_tables"],
                "n_columns": report["n_columns"],
                "n_missing": report["n_missing"],
                "n_collisions": report["n_collisions"],
                "out": args.out,
            },
            indent=2,
        )
    )
    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
