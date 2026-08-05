#!/usr/bin/env python3
"""Materialize WRDS enrichment into the mascotrl lake (link, ADV, Compustat).

Requires WRDS_USERNAME / WRDS_PW in env or gitignored .env.
Does not redistribute vendor data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mascotrl.data.paths import LAKE_ROOT  # noqa: E402
from mascotrl.data.wrds_enrich import materialize_wrds_enrichment  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--lake-dir",
        default=str(LAKE_ROOT),
        help="Lake root (default MASCOTRL_LAKE_DIR / mounted volsurf_data_lake)",
    )
    p.add_argument(
        "--skip-compustat",
        action="store_true",
        help="Only write CRSP-OM link + ADV (faster)",
    )
    args = p.parse_args()
    out = materialize_wrds_enrichment(
        args.lake_dir,
        use_local_crsp=True,
        fetch_compustat=not args.skip_compustat,
    )
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
