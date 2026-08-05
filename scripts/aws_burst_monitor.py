#!/usr/bin/env python3
"""Poll AWS Batch queues for a wave; write crash-survivable status JSON.

Usage::

    python scripts/aws_burst_monitor.py PICK --out logs/aws_burst_monitor_PICK.json
    python scripts/aws_burst_monitor.py PICK --once
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

from mascotrl.aws_burst.aws_client import BurstClient
from mascotrl.aws_burst.profiles import REGION, armed_profiles


STATUSES = (
    "SUBMITTED",
    "PENDING",
    "RUNNABLE",
    "STARTING",
    "RUNNING",
    "SUCCEEDED",
    "FAILED",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def snapshot_wave(
    root: Path,
    wave: str,
    *,
    profiles: list[str] | None = None,
    expect_succeeded: int | None = None,
    submission_job_ids: set[str] | None = None,
) -> dict[str, Any]:
    armed = armed_profiles(root)
    if profiles:
        want = set(profiles)
        armed = [p for p in armed if p["profile"] in want]
    # Scope to current wave submission when wave_*_submit.json exists.
    scoped_ids = submission_job_ids
    if scoped_ids is None:
        submit_path = root / f"deploy/aws_burst/config/wave_{wave}_submit.json"
        if submit_path.is_file():
            try:
                payload = json.loads(submit_path.read_text(encoding="utf-8"))
                scoped_ids = {
                    str(s["job_id"])
                    for s in (payload.get("submitted") or [])
                    if s.get("job_id")
                }
            except Exception:
                scoped_ids = None
    accounts: list[dict[str, Any]] = []
    totals = {s: 0 for s in STATUSES}
    failed: list[dict[str, Any]] = []
    for info in armed:
        profile = info["profile"]
        client = BurstClient(profile, REGION)
        by_status: dict[str, int] = {}
        for st in STATUSES:
            jobs = client.list_jobs("volsurf-burst-queue", st)
            # Filter to this wave by job name prefix when present.
            wave_jobs = [
                j
                for j in jobs
                if str(j.get("jobName") or "").startswith(f"mascotrl-{wave}-")
            ]
            if scoped_ids is not None:
                # Keep parent or array child whose id / parent is in this submit.
                filtered = []
                for j in wave_jobs:
                    jid = str(j.get("jobId") or "")
                    parent = jid.split(":")[0]
                    if jid in scoped_ids or parent in scoped_ids:
                        filtered.append(j)
                wave_jobs = filtered
            by_status[st] = len(wave_jobs)
            totals[st] += len(wave_jobs)
            if st == "FAILED":
                for j in wave_jobs[:20]:
                    failed.append(
                        {
                            "profile": profile,
                            "job_id": j.get("jobId"),
                            "job_name": j.get("jobName"),
                            "status_reason": j.get("statusReason"),
                        }
                    )
        accounts.append(
            {
                "profile": profile,
                "account_id": info.get("account_id"),
                "by_status": by_status,
            }
        )
    terminal = totals["SUCCEEDED"] + totals["FAILED"]
    active = sum(
        totals[s]
        for s in ("SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING")
    )
    # Prefer expected-success gate when provided (ignores historical FAILED noise).
    if expect_succeeded is not None:
        complete = active == 0 and totals["SUCCEEDED"] >= int(expect_succeeded)
    else:
        complete = active == 0 and totals["SUCCEEDED"] > 0 and totals["FAILED"] == 0
    return {
        "wave": wave,
        "polled_at": _utc(),
        "accounts": accounts,
        "totals": totals,
        "n_active": active,
        "n_terminal": terminal,
        "n_failed": totals["FAILED"],
        "n_succeeded": totals["SUCCEEDED"],
        "failed": failed,
        "complete": complete,
        "expect_succeeded": expect_succeeded,
        "scoped_to_submission": scoped_ids is not None,
        "n_scoped_job_ids": len(scoped_ids) if scoped_ids is not None else None,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("wave")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Status JSON path (default logs/aws_burst_monitor_<wave>.json).",
    )
    p.add_argument("--profiles", default="", help="Comma-separated profile filter.")
    p.add_argument("--once", action="store_true")
    p.add_argument("--poll-seconds", type=float, default=60.0)
    p.add_argument("--timeout-seconds", type=float, default=0.0)
    p.add_argument(
        "--expect-succeeded",
        type=int,
        default=None,
        help="Complete when SUCCEEDED >= N and no active jobs (ignores old FAILED).",
    )
    args = p.parse_args(argv)
    out = args.out or (ROOT / f"logs/aws_burst_monitor_{args.wave}.json")
    profiles = [x.strip() for x in str(args.profiles).split(",") if x.strip()] or None
    t0 = time.monotonic()
    while True:
        snap = snapshot_wave(
            ROOT,
            args.wave,
            profiles=profiles,
            expect_succeeded=args.expect_succeeded,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "polled_at": snap["polled_at"],
                    "n_active": snap["n_active"],
                    "n_succeeded": snap["n_succeeded"],
                    "n_failed": snap["n_failed"],
                    "complete": snap["complete"],
                }
            ),
            flush=True,
        )
        if args.once or snap["complete"]:
            return 0 if snap["n_failed"] == 0 else 1
        if args.timeout_seconds and (time.monotonic() - t0) > float(args.timeout_seconds):
            print("monitor_timeout", flush=True)
            return 2
        time.sleep(float(args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
