#!/usr/bin/env python3
"""After deep-resume RC6_NARRATIVE parents die, retarget resume digests and stem-submit.

Waits until no Batch parents remain for the original 10h wave (63d328a3) or
P0a (972f8066), then:
1. Rewrites resume/<stem>/image_digest.json to the new fleet digest
2. Stem-submits missing narrative finals with --pin-digest

Does not kill any jobs. Safe to re-run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aws_submit_wave import remaining_wall_attempt_seconds, submit_wave  # noqa: E402
from src.aws_burst.aws_client import BurstClient  # noqa: E402
from src.aws_burst.profiles import REGION, armed_profiles, artifact_bucket  # noqa: E402

JOB_QUEUE = "volsurf-burst-queue"
_ACTIVE = ("SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING")
DEEP_SHA_MARKERS = ("63d328a3", "972f8066")
LOG = ROOT / "logs" / "aws_burst" / "deadline_narrative_resume.log"
DIGEST_PATH = ROOT / "deploy" / "aws_burst" / "config" / "image_digest_volsurf-burst-1.json"

DEEP_STEMS = [
    "eq_K100_multi_happo_mlp_mean_std_cao",
    "eq_K100_single_cppo_mlp_softmax_mean_std_cao",
    "eq_K100_single_cppo_mlp_sparse_tilt_mean_std_cao",
    "eq_K100_single_ddpg_mlp_softmax_mtm_pnl",
    "eq_K100_single_ddpg_mlp_sparse_tilt_mtm_pnl",
    "eq_K100_single_ppo_mlp_softmax_differential_sharpe",
    "eq_K100_single_ppo_mlp_softmax_mean_std_cao",
    "eq_K100_single_ppo_mlp_softmax_mean_std_cao_uni-crucible",
    "eq_K100_single_ppo_mlp_sparse_tilt_differential_sharpe",
    "eq_K100_single_ppo_mlp_sparse_tilt_mean_std_cao",
    "eq_K100_single_ppo_mlp_sparse_tilt_mean_std_cao_uni-crucible",
]


def _log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def count_deep_narrative_parents(root: Path) -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
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
                if "RC6_NARRATIVE" not in name:
                    continue
                if any(m in name for m in DEEP_SHA_MARKERS):
                    found.append((prof["profile"], jid, name))
    return found


def new_digest() -> str:
    blob = json.loads(DIGEST_PATH.read_text(encoding="utf-8"))
    dig = str(blob.get("digest") or "").strip()
    if not dig.startswith("sha256:"):
        raise SystemExit(f"bad digest in {DIGEST_PATH}: {dig!r}")
    return dig


def rewrite_resume_digests(root: Path, digest: str) -> int:
    n = 0
    body = json.dumps(
        {"image_digest": digest, "container_digest": digest}, indent=2
    ).encode("utf-8")
    for prof in armed_profiles(root):
        client = BurstClient(prof["profile"], REGION)
        bucket = artifact_bucket(client.account_id())
        s3 = client._s3()
        for stem in DEEP_STEMS:
            prefix = f"RC6_NARRATIVE/resume/{stem}/"
            # Only rewrite if resume prefix has objects (real progress).
            keys = list(client.list_keys(bucket, prefix))
            if not keys:
                continue
            key = f"{prefix}image_digest.json"
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
            )
            n += 1
            _log(f"rewrote {prof['profile']} {key}")
    return n


def missing_stems(root: Path) -> list[str]:
    missing: list[str] = []
    for stem in DEEP_STEMS:
        found = False
        for prof in armed_profiles(root):
            client = BurstClient(prof["profile"], REGION)
            bucket = artifact_bucket(client.account_id())
            if client.head_exists(bucket, f"RC6_NARRATIVE/{stem}.json"):
                found = True
                break
        if not found:
            missing.append(stem)
    return missing


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--poll-seconds", type=float, default=120.0)
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    digest = new_digest()
    _log(f"target_digest={digest}")

    while True:
        parents = count_deep_narrative_parents(ROOT)
        if parents:
            _log(f"waiting deep_narrative_parents={len(parents)}")
            for prof, jid, name in parents[:8]:
                _log(f"  still {prof} {jid[:8]} {name[:70]}")
            if args.once:
                return 2
            time.sleep(float(args.poll_seconds))
            continue

        _log("deep narrative parents gone; rewriting resume digests")
        if args.dry_run:
            _log("dry-run: skip rewrite/submit")
            return 0
        n = rewrite_resume_digests(ROOT, digest)
        _log(f"rewrote_n={n}")

        missing = missing_stems(ROOT)
        _log(f"missing_finals={missing}")
        if not missing:
            _log("all deep narrative finals present; done")
            return 0

        try:
            attempt_s = remaining_wall_attempt_seconds()
        except ValueError as exc:
            _log(f"stop: {exc}")
            return 3

        result = submit_wave(
            ROOT,
            "RC6_NARRATIVE",
            dry_run=False,
            stems=missing,
            skip_spend_gate=True,
            force=True,
            attempt_seconds=attempt_s,
            pin_digest=digest,
        )
        out = ROOT / "logs" / "aws_burst" / "deadline_narrative_resume_submit.json"
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        _log(f"submitted missing={len(missing)} log={out}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
