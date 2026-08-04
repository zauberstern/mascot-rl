#!/usr/bin/env python3
"""Overlay LSEG P0/P2 onto the existing parquet lake. Never overwrite OM/CRSP SoT."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.lseg_overlay import ingest_lseg_overlays  # noqa: E402
from src.data.paths import LSEG_RAW, LAKE_ROOT, assert_lake_mounted, assert_raw_mounted  # noqa: E402
from src.logging_utils import setup_logging  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lseg-raw", default="")
    p.add_argument("--lake", default=str(LAKE_ROOT))
    args = p.parse_args()
    log = setup_logging(log_file=str(ROOT / "logs" / "ingest_lseg_overlays.log"))
    raw = Path(args.lseg_raw) if args.lseg_raw else Path(LSEG_RAW)
    assert_raw_mounted(raw.parent if raw.name == "lseg" else raw)
    lake = Path(args.lake)
    assert_lake_mounted(lake)
    if not raw.is_dir():
        log.error("lseg raw missing: %s", raw)
        sys.exit(1)
    lake_macro = lake / "macro"
    info = ingest_lseg_overlays(lseg_raw=raw, lake_macro=lake_macro)
    info["lseg_raw"] = str(raw)
    info["lake_macro"] = str(lake_macro)
    log.info("lseg overlay done %s", info)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
