#!/usr/bin/env python3
"""Wait for WRDS auth, then catalog + materialize under ``macro/wrds/``.

Layout (sustainable):
  {LAKE_ROOT}/macro/wrds/          full Compustat / IBES dumps + provenance
  {LAKE_ROOT}/macro/*.parquet      S5 spine macros (thin enrich regenerated in place)

Does not touch Tier A/B hive panels or Downloads dumps.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.paths import LAKE_ROOT, MASCOTRL_ROOT  # noqa: E402
from src.data.wrds_enrich import _load_dotenv_files, connect_wrds  # noqa: E402
from src.logging_utils import setup_logging  # noqa: E402

STATUS = MASCOTRL_ROOT / "logs" / "artifacts" / "wrds_campaign_status.json"
README = """# WRDS enrichments (lake)

Sustainable layout for vendor pulls. Do not scatter Compustat/IBES into hive panel dirs.

| Path | Role |
|---|---|
| `comp_company.parquet` | Compustat company master |
| `comp_ccm_link.parquet` | CRSP/Compustat CCM link history |
| `comp_funda_full.parquet` | Full annual BS/IS/CF (all entitled cols) |
| `comp_fundq_full.parquet` | Full quarterly fundamentals |
| `comp_funda_spine.parquet` | Thin projection twin of spine enrich |
| `ibes_*.parquet` | IBES tables from entitlement campaign |
| `*_provenance.json` | Pull timestamps, counts, column lists |

Spine contract (S5): keep thin `../compustat_funda_enrich.parquet` and existing `om_*` /
`crsp_*` macros at `macro/` root. Full dumps live **only** here under `wrds/`.

Redistribution: WRDS license forbids shipping these files. Local lake only.
"""


def _write_status(payload: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    STATUS.write_text(json.dumps(payload, indent=2) + "\n")


def _run(cmd: list[str], log) -> int:
    log.info("run: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return int(proc.returncode)


def wait_connect(*, attempts: int, sleep_s: float, log) -> None:
    _load_dotenv_files()
    last_err = ""
    for i in range(1, attempts + 1):
        _write_status(
            {
                "phase": "wait_auth",
                "attempt": i,
                "attempts": attempts,
                "ok": False,
                "hint": (
                    "Approve Duo Push if it arrives. If PAM fails in <5s with no push: "
                    "password/account rejected, or login to WRDS website from this IPv4 "
                    "first (30-day IP session) and enable Duo Push."
                ),
            }
        )
        log.info("connect attempt %d/%d — approve Duo Push if notified", i, attempts)
        try:
            conn = connect_wrds()
            n = len(conn.list_libraries())
            conn.close()
            log.info("CONNECT_OK libraries=%d", n)
            _write_status({"phase": "connected", "ok": True, "n_libraries": n})
            return
        except Exception as exc:
            last_err = str(exc)[:500]
            log.warning("connect failed: %s", last_err[:200])
            _write_status(
                {
                    "phase": "wait_auth",
                    "attempt": i,
                    "ok": False,
                    "error": last_err,
                }
            )
            if i < attempts:
                time.sleep(sleep_s)
    raise EnvironmentError(f"WRDS auth not recovered after {attempts} attempts: {last_err}")


def ensure_layout(lake: Path) -> Path:
    wrds_dir = lake / "macro" / "wrds"
    wrds_dir.mkdir(parents=True, exist_ok=True)
    readme = wrds_dir / "README.md"
    if not readme.is_file():
        readme.write_text(README)
    return wrds_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lake-dir", default=None)
    p.add_argument("--attempts", type=int, default=40)
    p.add_argument("--sleep-s", type=float, default=45.0)
    p.add_argument("--year-start", type=int, default=2003)
    p.add_argument("--year-end", type=int, default=2024)
    p.add_argument("--skip-ibes", action="store_true")
    p.add_argument("--skip-seal", action="store_true")
    p.add_argument("--connect-only", action="store_true")
    p.add_argument(
        "--log-file",
        default=str(MASCOTRL_ROOT / "logs" / "wrds_enrich_campaign.log"),
    )
    args = p.parse_args()
    log = setup_logging(log_file=args.log_file)
    lake = Path(args.lake_dir) if args.lake_dir else LAKE_ROOT
    wrds_dir = ensure_layout(lake)
    log.info("lake=%s wrds_dir=%s", lake, wrds_dir)

    wait_connect(attempts=args.attempts, sleep_s=args.sleep_s, log=log)
    if args.connect_only:
        sys.exit(0)

    py = str(ROOT / ".venv" / "bin" / "python")
    steps = [
        ([py, "scripts/wrds_catalog_entitlements.py"], "catalog"),
        (
            [
                py,
                "scripts/materialize_wrds_fundamentals.py",
                "--lake-dir",
                str(lake),
                "--year-start",
                str(args.year_start),
                "--year-end",
                str(args.year_end),
            ],
            "fundamentals",
        ),
    ]
    if not args.skip_ibes:
        steps.append(
            ([py, "scripts/materialize_wrds_ibes.py", "--lake-dir", str(lake)], "ibes")
        )
    if not args.skip_seal:
        steps.append(
            (
                [
                    py,
                    "scripts/audit_lake_source_coverage.py",
                    "--skip-heavy-count-only",
                    "--wrds",
                ],
                "seal_s5",
            )
        )

    for cmd, name in steps:
        _write_status({"phase": name, "ok": False})
        rc = _run(cmd, log)
        if rc != 0:
            _write_status({"phase": name, "ok": False, "exit_code": rc})
            log.error("step %s failed rc=%d", name, rc)
            sys.exit(rc)
        _write_status({"phase": name, "ok": True, "exit_code": 0})

    _write_status(
        {
            "phase": "done",
            "ok": True,
            "wrds_dir": str(wrds_dir),
            "note": "Dump delete only after seal verdict COMPLETE",
        }
    )
    log.info("campaign done wrds_dir=%s", wrds_dir)
    sys.exit(0)


if __name__ == "__main__":
    main()
