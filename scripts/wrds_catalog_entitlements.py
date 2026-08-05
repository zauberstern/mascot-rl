#!/usr/bin/env python3
"""Catalog every WRDS library/table visible to this account."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mascotrl.data.wrds_catalog import build_entitlement_catalog  # noqa: E402
from mascotrl.data.wrds_enrich import _load_dotenv_files, connect_wrds  # noqa: E402
from mascotrl.data.paths import MASCOTRL_ROOT  # noqa: E402
from mascotrl.logging_utils import setup_logging  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        default=str(MASCOTRL_ROOT / "logs" / "artifacts" / "wrds_entitlement_catalog.json"),
    )
    p.add_argument("--log-file", default=str(MASCOTRL_ROOT / "logs" / "wrds_catalog.log"))
    args = p.parse_args()
    log = setup_logging(log_file=args.log_file)
    _load_dotenv_files()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = connect_wrds()
    except Exception as exc:
        payload = {
            "ok": False,
            "error": str(exc)[:500],
            "note": "Fix WRDS PAM/password/IP allowlist then re-run",
        }
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        log.error("catalog blocked: %s", exc)
        sys.exit(2)
    try:
        catalog = build_entitlement_catalog(conn)
        catalog["ok"] = True
        out.write_text(json.dumps(catalog, indent=2, default=str) + "\n", encoding="utf-8")
        log.info(
            "wrote %s n_libs=%s n_tables=%s classes=%s",
            out,
            catalog.get("n_libraries"),
            catalog.get("n_tables"),
            catalog.get("class_counts"),
        )
    finally:
        conn.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
