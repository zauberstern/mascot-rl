#!/usr/bin/env python3
"""Pull IBES (and other catalog FUNDAMENTALS/IBES gaps) not already in the lake.

Reads ``logs/artifacts/wrds_entitlement_catalog.json`` when present; otherwise
attempts a live catalog. Writes under ``macro/wrds/``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.paths import LAKE_ROOT, MASCOTRL_ROOT  # noqa: E402
from src.data.wrds_enrich import _load_dotenv_files, connect_wrds  # noqa: E402
from src.logging_utils import setup_logging  # noqa: E402

# Prefer a small set of high-value IBES tables if entitled.
IBES_PREFERRED = (
    "ibes.statsumu",
    "ibes.actu_epsus",
    "ibes.detu_epsus",
    "ibes.idsum",
)


def _safe_table_name(full: str) -> str:
    return full.replace(".", "_").replace("/", "_")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lake-dir", default=None)
    p.add_argument(
        "--catalog",
        default=str(MASCOTRL_ROOT / "logs" / "artifacts" / "wrds_entitlement_catalog.json"),
    )
    p.add_argument("--log-file", default=str(MASCOTRL_ROOT / "logs" / "wrds_ibes_pull.log"))
    args = p.parse_args()
    log = setup_logging(log_file=args.log_file)
    _load_dotenv_files()
    lake = Path(args.lake_dir) if args.lake_dir else LAKE_ROOT
    out_dir = lake / "macro" / "wrds"
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = Path(args.catalog)
    ibes_tables: list[str] = []
    if catalog_path.is_file():
        cat = json.loads(catalog_path.read_text(encoding="utf-8"))
        if not cat.get("ok", True) and cat.get("error"):
            log.error("catalog not ok: %s", cat.get("error"))
            sys.exit(2)
        ibes_tables = [
            r["table"]
            for r in cat.get("tables") or []
            if r.get("class") == "IBES"
        ]
    if not ibes_tables:
        ibes_tables = list(IBES_PREFERRED)

    # Prefer preferred order, then any extras from catalog
    ordered: list[str] = []
    for t in IBES_PREFERRED:
        if t in ibes_tables and t not in ordered:
            ordered.append(t)
    for t in ibes_tables:
        if t not in ordered:
            ordered.append(t)

    try:
        conn = connect_wrds()
    except Exception as exc:
        log.error("WRDS connect failed: %s", exc)
        sys.exit(2)

    written: dict[str, str] = {}
    try:
        for full in ordered[:12]:  # cap first campaign
            schema, _, table = full.partition(".")
            if not schema or not table:
                continue
            dest = out_dir / f"{_safe_table_name(full)}.parquet"
            if dest.is_file() and dest.stat().st_size > 1000:
                log.info("reuse %s", dest.name)
                written[full] = str(dest)
                continue
            log.info("pull %s …", full)
            try:
                df = conn.raw_sql(f"select * from {schema}.{table}")
            except Exception as exc:
                log.warning("skip %s: %s", full, exc)
                continue
            if df is None or len(df) == 0:
                log.warning("empty %s", full)
                continue
            df.to_parquet(dest, index=False)
            written[full] = str(dest)
            log.info("wrote %s rows=%d", dest.name, len(df))
    finally:
        conn.close()

    prov = out_dir / "wrds_ibes_provenance.json"
    prov.write_text(json.dumps({"paths": written, "n": len(written)}, indent=2) + "\n")
    log.info("done n=%d provenance=%s", len(written), prov)
    sys.exit(0 if written else 1)


if __name__ == "__main__":
    main()
