#!/usr/bin/env python3
"""CLI: build AWS Burst panel bundle (Arctic + feature-cube lake files)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mascotrl.aws_burst.panel_bundle import build_panel_bundle  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "logs" / "aws_burst_panel_bundle",
    )
    p.add_argument(
        "--lake",
        type=Path,
        default=Path(
            os.environ.get(
                "MASCOTRL_LAKE_BASE",
                "/mnt/volsurf/volsurf_parquet_lake",
            )
        ),
    )
    p.add_argument(
        "--arctic",
        type=Path,
        default=Path(
            os.environ.get("MASCOTRL_ARCTIC_DIR", str(ROOT / "data" / "volsurf_arcticdb"))
        ),
    )
    p.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Do not refuse when some feature families are missing (dev only).",
    )
    args = p.parse_args(argv)
    meta = build_panel_bundle(
        lake=args.lake,
        arctic=args.arctic if args.arctic.is_dir() else None,
        out_dir=args.out,
        require_complete=not bool(args.allow_incomplete),
        repo_root=ROOT,
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
