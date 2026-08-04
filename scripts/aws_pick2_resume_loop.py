#!/usr/bin/env python3
"""Resume-loop driver for PICK2 narrative cells (12h Batch attempts + S3 resume).

Polls wave-root finals on all armed accounts; force-resubmits incomplete stems
until every expected cell has a validated final or wall-clock cap is hit.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aws_submit_wave import submit_wave  # noqa: E402
from src.aws_burst.aws_client import BurstClient
from src.aws_burst.profiles import REGION, armed_profiles, artifact_bucket
from src.aws_burst.waves import discover_wave_cells

DEFAULT_WALL_CLOCK = datetime(2026, 8, 28, 12, 0, 0, tzinfo=timezone.utc)
LOG_PATH = ROOT / "logs" / "campaign_sprint" / "pick2_resume_loop.log"
JOB_QUEUE = "volsurf-burst-queue"
_ACTIVE_BATCH = ("SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING")


def _log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def _expected_stems(root: Path) -> list[str]:
    manifest = root / "config/spectrum/cherrypick_final/manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        pick2 = data.get("pick2") or {}
        cells = pick2.get("cells")
        if cells:
            return list(cells)
    return [Path(c).stem for c in discover_wave_cells(root, "PICK2")]


def count_active_pick2_jobs(root: Path) -> int:
    """Count in-flight Batch parents whose name includes PICK2.

    The resume loop must not force-resubmit while prior attempts are still
    RUNNING/RUNNABLE: duplicate arrays thrash Spot capacity and can SIGTERM
    healthy children before they write finals.
    """
    n = 0
    for prof in armed_profiles(root):
        client = BurstClient(prof["profile"], REGION)
        for status in _ACTIVE_BATCH:
            try:
                jobs = client.list_jobs(JOB_QUEUE, status)
            except Exception:
                continue
            for job in jobs:
                jid = str(job.get("jobId") or "")
                if ":" in jid:
                    continue  # array child; parent already counted
                name = str(job.get("jobName") or "")
                if "PICK2" in name or "pick2" in name:
                    n += 1
    return n


def _resume_progress(client: BurstClient, stem: str) -> int:
    """Count resume manifest completed cells (fold progress proxy)."""
    bucket = artifact_bucket(client.account_id())
    prefix = f"PICK2/resume/{stem}/cpcv/"
    n = 0
    for key in client.list_keys(bucket, prefix):
        if key.endswith("campaign_manifest.json"):
            try:
                blob = client.get_json(bucket, key)
                n = max(n, len(blob.get("completed") or {}))
            except Exception:
                continue
    return n


def poll_status(root: Path, *, include_resume_progress: bool = False) -> dict[str, Any]:
    want = set(_expected_stems(root))
    found: set[str] = set()
    errors: list[str] = []
    progress: dict[str, int] = {}
    for prof in armed_profiles(root):
        client = BurstClient(prof["profile"], REGION)
        bucket = artifact_bucket(client.account_id())
        # Head expected stems only. Full Prefix=PICK2/ walks grow with
        # _archive_/resume objects and can hang the resume loop for minutes.
        for stem in sorted(want):
            final_key = f"PICK2/{stem}.json"
            err_key = f"PICK2/{stem}.error.json"
            if client.head_exists(bucket, final_key):
                found.add(stem)
            if client.head_exists(bucket, err_key):
                errors.append(f"{prof['profile']}:{stem}.error.json")
        if include_resume_progress:
            for stem in sorted(want - found):
                progress[stem] = max(
                    progress.get(stem, 0), _resume_progress(client, stem)
                )
    missing = sorted(want - found)
    return {
        "n_expected": len(want),
        "n_found": len(found),
        "missing": missing,
        "errors": errors,
        "resume_fold_counts": progress,
        "complete": not missing,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--poll-seconds", type=float, default=1800.0)
    p.add_argument(
        "--wall-clock-until",
        default=DEFAULT_WALL_CLOCK.isoformat(),
        help="ISO8601 UTC stop resubmitting after this time",
    )
    p.add_argument("--once", action="store_true", help="Single poll + maybe submit, then exit")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    wall = datetime.fromisoformat(str(args.wall_clock_until).replace("Z", "+00:00"))
    if wall.tzinfo is None:
        wall = wall.replace(tzinfo=timezone.utc)

    while True:
        status = poll_status(ROOT)
        _log(
            f"poll found={status['n_found']}/{status['n_expected']} "
            f"missing={status['missing'][:5]}{'...' if len(status['missing']) > 5 else ''} "
            f"errors={len(status['errors'])}"
        )
        if status["complete"]:
            _log("PICK2 complete")
            done = ROOT / "logs" / "campaign_sprint" / "pick2_complete.done"
            done.parent.mkdir(parents=True, exist_ok=True)
            done.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
            return 0

        now = datetime.now(timezone.utc)
        if now >= wall:
            _log(f"wall_clock cap reached; still missing {status['missing']}")
            return 3

        if not args.dry_run:
            active = count_active_pick2_jobs(ROOT)
            if active > 0:
                _log(f"skip_submit active_batch_jobs={active}")
            else:
                try:
                    result = submit_wave(ROOT, "PICK2", force=True, dry_run=False)
                    submitted = result.get("submitted") or []
                    _log(f"submit {json.dumps(submitted)}")
                except Exception as exc:
                    _log(f"submit_failed: {exc}")

        if args.once:
            return 1 if not status["complete"] else 0

        time.sleep(float(args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
