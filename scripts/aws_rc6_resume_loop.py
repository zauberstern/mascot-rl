#!/usr/bin/env python3
"""Resume-loop for RC6 narrative / HAPPO / revive stems until Mon 18:00 CEST.

Unlike PICK2's force=True full-wave resubmit, this loop:
1. Counts active Batch parents for each wave (skips array children with ':').
2. Sleeps while any parent is active.
3. Stem-filters only missing finals with no active child.
4. Uses remaining-wall attempt timeout (never 48h).
5. Pins resume digests via --pin-digest when an image_digest.json exists.
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

from scripts.aws_submit_wave import (  # noqa: E402
    remaining_wall_attempt_seconds,
    submit_wave,
)
from src.aws_burst.aws_client import BurstClient  # noqa: E402
from src.aws_burst.profiles import REGION, armed_profiles, artifact_bucket  # noqa: E402
from src.aws_burst.waves import discover_wave_cells  # noqa: E402

JOB_QUEUE = "volsurf-burst-queue"
_ACTIVE = ("SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING")
DEFAULT_WAVES = ("RC6_NARRATIVE", "RC6_HAPPO", "RC6", "RC6_K200", "RC6_HEADS")
LOG_PATH = ROOT / "logs" / "aws_burst" / "rc6_resume_loop.log"


def _log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def count_active_wave_parents(root: Path, wave: str) -> int:
    """Count in-flight Batch parents whose name starts with mascotrl-{wave}-."""
    n = 0
    prefix = f"mascotrl-{wave}-"
    for prof in armed_profiles(root):
        client = BurstClient(prof["profile"], REGION)
        for status in _ACTIVE:
            try:
                jobs = client.list_jobs(JOB_QUEUE, status)
            except Exception:
                continue
            for job in jobs:
                jid = str(job.get("jobId") or "")
                if ":" in jid:
                    continue
                name = str(job.get("jobName") or "")
                if name.startswith(prefix):
                    n += 1
    return n


def _expected_stems(root: Path, wave: str) -> list[str]:
    # Always use on-disk wave glob so stem-filtered revive submits that rewrite
    # wave_*_manifest.json cannot shrink the watched set.
    return [Path(c).stem for c in discover_wave_cells(root, wave)]


def _resume_digest(client: BurstClient, wave: str, stem: str) -> str | None:
    bucket = artifact_bucket(client.account_id())
    key = f"{wave}/resume/{stem}/image_digest.json"
    try:
        blob = client.get_json(bucket, key)
    except Exception:
        return None
    dig = str(blob.get("image_digest") or blob.get("container_digest") or "").strip()
    return dig or None


def poll_wave(root: Path, wave: str) -> dict[str, Any]:
    want = set(_expected_stems(root, wave))
    found: set[str] = set()
    errors: list[str] = []
    digests: dict[str, str] = {}
    for prof in armed_profiles(root):
        client = BurstClient(prof["profile"], REGION)
        bucket = artifact_bucket(client.account_id())
        for stem in sorted(want):
            if client.head_exists(bucket, f"{wave}/{stem}.json"):
                found.add(stem)
            if client.head_exists(bucket, f"{wave}/{stem}.error.json"):
                errors.append(f"{prof['profile']}:{stem}")
            dig = _resume_digest(client, wave, stem)
            if dig:
                digests[stem] = dig
    missing = sorted(want - found)
    return {
        "wave": wave,
        "n_expected": len(want),
        "n_found": len(found),
        "missing": missing,
        "errors": errors,
        "resume_digests": digests,
        "complete": not missing,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--waves",
        default=",".join(DEFAULT_WAVES),
        help="Comma-separated waves to watch/resume",
    )
    p.add_argument("--poll-seconds", type=float, default=600.0)
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--stems",
        default="",
        help="Optional comma-separated stem allowlist (across all waves).",
    )
    args = p.parse_args(argv)
    waves = [w.strip() for w in str(args.waves).split(",") if w.strip()]
    allow = {s.strip() for s in str(args.stems).split(",") if s.strip()} or None

    while True:
        try:
            attempt_s = remaining_wall_attempt_seconds()
        except ValueError as exc:
            _log(f"stop: {exc}")
            return 3

        any_incomplete = False
        for wave in waves:
            status = poll_wave(ROOT, wave)
            missing = status["missing"]
            if allow is not None:
                missing = [s for s in missing if s in allow]
            _log(
                f"{wave} found={status['n_found']}/{status['n_expected']} "
                f"missing={missing[:5]}{'...' if len(missing) > 5 else ''} "
                f"errors={len(status['errors'])} attempt_s={attempt_s}"
            )
            if not missing:
                continue
            any_incomplete = True
            active = count_active_wave_parents(ROOT, wave)
            if active > 0:
                _log(f"{wave} skip_submit active_batch_parents={active}")
                continue
            # Group missing by resume digest so pin-digest submits stay coherent.
            by_digest: dict[str | None, list[str]] = {}
            for stem in missing:
                dig = status["resume_digests"].get(stem)
                by_digest.setdefault(dig, []).append(stem)
            for dig, stems in by_digest.items():
                if args.dry_run:
                    _log(f"{wave} dry_run stems={stems} pin={dig}")
                    continue
                try:
                    result = submit_wave(
                        ROOT,
                        wave,
                        force=True,
                        skip_spend_gate=True,
                        stems=stems,
                        attempt_seconds=attempt_s,
                        pin_digest=dig,
                    )
                    submitted = result.get("submitted") or []
                    _log(
                        f"{wave} submit pin={dig} "
                        f"{json.dumps([{k: s.get(k) for k in ('profile', 'n', 'skipped')} for s in submitted])}"
                    )
                except Exception as exc:
                    _log(f"{wave} submit_failed: {exc}")

        if not any_incomplete:
            _log("all watched waves complete")
            return 0
        if args.once:
            return 1
        time.sleep(float(args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
