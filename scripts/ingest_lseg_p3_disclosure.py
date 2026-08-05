#!/usr/bin/env python3
"""Copy LSEG P3 macro parquets into the lake for disclosure only.

Keeps ``P3_REFUSED`` in ``src/data/lseg_overlay.py`` unchanged. Files land under
``{lake}/macro/p3/`` with ``_provenance.json`` stating not feature-admitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mascotrl.data.paths import LAKE_ROOT, LSEG_RAW, RAW_ROOT, assert_lake_mounted, assert_raw_mounted
from mascotrl.logging_utils import setup_logging

P3_FILES = (
    "lseg_ibes.parquet",
    "lseg_short_interest.parquet",
    "lseg_worldscope.parquet",
)

DEFAULT_SRC_DIR = LSEG_RAW / "macro" / "p3"


def _sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ingest_lseg_p3_disclosure(
    *,
    src_dir: Path,
    lake: Path,
) -> dict:
    dest_dir = lake / "macro" / "p3"
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict] = []
    missing: list[str] = []
    for name in P3_FILES:
        src = src_dir / name
        if not src.is_file():
            missing.append(name)
            continue
        dest = dest_dir / name
        shutil.copy2(src, dest)
        copied.append(
            {
                "name": name,
                "source": str(src),
                "dest": str(dest),
                "bytes": dest.stat().st_size,
                "sha256": _sha256(dest),
            }
        )

    prov = {
        "role": "disclosure-only",
        "feature_admitted": False,
        "overlay_refused": True,
        "note": (
            "disclosure-only, not feature-admitted; "
            "P3_REFUSED in src/data/lseg_overlay.py remains unchanged"
        ),
        "source_dir": str(src_dir),
        "dest_dir": str(dest_dir),
        "copied": copied,
        "missing": missing,
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (dest_dir / "_provenance.json").write_text(
        json.dumps(prov, indent=2) + "\n", encoding="utf-8"
    )
    if missing:
        raise FileNotFoundError(f"P3 sources missing under {src_dir}: {missing}")
    return prov


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-dir", default=str(DEFAULT_SRC_DIR))
    p.add_argument("--lake", default=str(LAKE_ROOT))
    args = p.parse_args()
    log = setup_logging(log_file=str(ROOT / "logs" / "ingest_lseg_p3_disclosure.log"))
    assert_raw_mounted(RAW_ROOT)
    lake = assert_lake_mounted(Path(args.lake))
    info = ingest_lseg_p3_disclosure(src_dir=Path(args.src_dir), lake=lake)
    log.info("LSEG P3 disclosure ingest done n=%d", len(info.get("copied") or []))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
