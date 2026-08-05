#!/usr/bin/env python3
"""Ingest non-contract WRDS IBES financial ratios into the lake (disclosure-only).

Writes ``{lake}/macro/ibes_financial_ratios.parquet`` (ZSTD). Does not admit
features into the obs cube. Source remains non-contract / not redistributable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mascotrl.data.paths import LAKE_ROOT, RAW_ROOT, assert_lake_mounted, assert_raw_mounted
from mascotrl.logging_utils import setup_logging

DEFAULT_SRC = (
    RAW_ROOT / "non_contract" / "wrds_ibes_financial_ratios_sp500.csv"
)
DEFAULT_DEST_NAME = "ibes_financial_ratios.parquet"


def _sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ingest_ibes(
    *,
    src: Path,
    lake: Path,
    compression: str = "zstd",
) -> dict:
    import duckdb

    if not src.is_file():
        raise FileNotFoundError(f"IBES source missing: {src}")
    macro = lake / "macro"
    macro.mkdir(parents=True, exist_ok=True)
    dest = macro / DEFAULT_DEST_NAME
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.is_file():
        tmp.unlink()

    con = duckdb.connect()
    try:
        src_s = src.as_posix().replace("'", "''")
        tmp_s = tmp.as_posix().replace("'", "''")
        con.execute(
            f"""
            COPY (
              SELECT * FROM read_csv_auto('{src_s}', header=true, ignore_errors=true)
            ) TO '{tmp_s}' (FORMAT PARQUET, COMPRESSION '{compression}')
            """
        )
        n = int(con.execute(f"SELECT COUNT(*) FROM read_parquet('{tmp_s}')").fetchone()[0])
    finally:
        con.close()

    tmp.replace(dest)
    prov = {
        "source": str(src),
        "dest": str(dest),
        "n_rows": n,
        "source_sha256": _sha256(src),
        "source_bytes": src.stat().st_size,
        "dest_bytes": dest.stat().st_size,
        "compression": compression,
        "role": "disclosure-only",
        "feature_admitted": False,
        "non_contract": True,
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Non-contract IBES ratios; not wired into obs cube this cycle.",
    }
    (macro / "ibes_financial_ratios_provenance.json").write_text(
        json.dumps(prov, indent=2) + "\n", encoding="utf-8"
    )
    return prov


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default=str(DEFAULT_SRC))
    p.add_argument("--lake", default=str(LAKE_ROOT))
    args = p.parse_args()
    log = setup_logging(log_file=str(ROOT / "logs" / "ingest_ibes_noncontract.log"))
    assert_raw_mounted(Path(args.src).parents[1] if "non_contract" in Path(args.src).parts else RAW_ROOT)
    lake = assert_lake_mounted(Path(args.lake))
    info = ingest_ibes(src=Path(args.src), lake=lake)
    log.info("IBES ingest done %s", info)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
