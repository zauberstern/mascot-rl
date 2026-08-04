#!/usr/bin/env python3
"""Materialize lake feature panels to parquet mirrors + ArcticDB."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.feature_panels import materialize_feature_panels  # noqa: E402
from src.data.paths import ARCTIC_ROOT, LAKE_ROOT, assert_lake_mounted  # noqa: E402
from src.logging_utils import setup_logging  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2003-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--lake", default=str(LAKE_ROOT))
    p.add_argument("--arctic", default=str(ARCTIC_ROOT))
    p.add_argument("--out-dir", default="")
    p.add_argument("--no-arctic", action="store_true")
    p.add_argument("--families", default="", help="Comma-separated family subset")
    args = p.parse_args()
    log = setup_logging(log_file=str(ROOT / "logs" / "materialize_feature_panels.log"))
    lake = assert_lake_mounted(Path(args.lake))
    families = [x.strip() for x in args.families.split(",") if x.strip()] or None
    info = materialize_feature_panels(
        start=args.start,
        end=args.end,
        lake=lake,
        arctic_path=args.arctic,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        families=families,
        persist_arctic=not args.no_arctic,
    )
    log.info("done families=%s", list((info.get("families") or {}).keys()))
    print(json.dumps({k: v for k, v in info.items() if k != "families"}, indent=2, default=str))
    print(json.dumps({k: {"rows": v.get("rows"), "nan_rates": v.get("nan_rates")} for k, v in (info.get("families") or {}).items()}, indent=2))


if __name__ == "__main__":
    main()
