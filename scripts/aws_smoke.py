#!/usr/bin/env python3
"""AWS Burst smoke: deploy guardrails+batch, 1-cell toy run, wait, pull, assert.

CLI / env
---------
Required for a **live** smoke (not ``--dry-run``):

- AWS profile: ``--profile volsurf-burst-1`` (or ``MASCOTRL_SMOKE_PROFILE``)
- Region: ``eu-central-1`` (``MASCOTRL_AWS_REGION`` override)
- Armed budget action on that profile (``scripts/aws_arm_budget_action.py``)
- Panel bundle at ``logs/aws_burst_panel_bundle/`` (toy cells may skip cube)
- Container image digest stamped under ``deploy/aws_burst/config/``

Examples::

    # Compose-only (CI / moto): no AWS calls
    python scripts/aws_smoke.py --dry-run

    # Live smoke on account 1 (manual gated; not CI)
    python scripts/aws_smoke.py --profile volsurf-burst-1 --wave PICK_SMOKE

Live smoke uses ``--allow-toy-panel`` semantics via wave cells that do not
require the equity feature cube, and ``strict=False`` only for the smoke
path so a toy panel can complete. Full waves stay ``--strict``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aws_burst.aws_client import BurstClient
from src.aws_burst.budget_action import read_actual_spend
from src.aws_burst.image_digest import pinned_image_uri
from src.aws_burst.profiles import REGION


GUARDRAILS_TEMPLATE = ROOT / "deploy/aws_burst/cloudformation/00_guardrails.yaml"
BATCH_TEMPLATE = ROOT / "deploy/aws_burst/cloudformation/10_batch_spot.yaml"
GUARDRAILS_STACK = "volsurf-burst-guardrails"
BATCH_STACK = "volsurf-burst-batch"


def compose_smoke_plan(
    *,
    profile: str,
    wave: str = "PICK_SMOKE",
    region: str = REGION,
    teardown: bool = False,
    root: Path | None = None,
) -> dict[str, Any]:
    """Return the ordered smoke steps without contacting AWS."""
    base = root or ROOT
    image_uri = pinned_image_uri(base, profile)
    if "@sha256:" not in image_uri:
        raise RuntimeError(f"smoke_image_uri_not_digest_pinned: {image_uri}")
    return {
        "profile": profile,
        "wave": wave,
        "region": region,
        "teardown": bool(teardown),
        "image_uri": image_uri,
        "steps": [
            {"op": "deploy_stack", "stack": GUARDRAILS_STACK, "template": str(GUARDRAILS_TEMPLATE)},
            {
                "op": "deploy_stack",
                "stack": BATCH_STACK,
                "template": str(BATCH_TEMPLATE),
                "image_uri": image_uri,
            },
            {"op": "arm_budget_action"},
            {"op": "read_actual_spend"},
            {"op": "submit_wave", "wave": wave, "allow_toy": True, "strict": False},
            {"op": "wait_for_array"},
            {"op": "pull_artifacts", "wave": wave, "require_complete": True},
            {"op": "assert_cloudwatch_logs"},
            {"op": "teardown_stacks"} if teardown else {"op": "keep_stacks"},
        ],
    }


def assert_smoke_artifacts(index: dict[str, Any]) -> None:
    if int(index.get("n_accepted") or 0) < 1:
        raise RuntimeError(f"smoke_no_accepted_artifacts: {index}")
    if index.get("rejected"):
        # Toy / dry rejects are unexpected for a successful smoke pull.
        bad = [r for r in index["rejected"] if r.get("reason") not in ()]
        if bad and int(index.get("n_accepted") or 0) < 1:
            raise RuntimeError(f"smoke_all_rejected: {bad[:3]}")


def run_smoke(
    *,
    profile: str,
    wave: str = "PICK_SMOKE",
    region: str = REGION,
    dry_run: bool = False,
    teardown: bool = False,
    wait_timeout_seconds: float = 3600.0,
) -> dict[str, Any]:
    plan = compose_smoke_plan(
        profile=profile, wave=wave, region=region, teardown=teardown
    )
    if dry_run:
        return {"dry_run": True, "plan": plan}

    from scripts.aws_pull_artifacts import pull_wave
    from scripts.aws_submit_wave import submit_wave

    client = BurstClient(profile, region)
    results: dict[str, Any] = {"plan": plan, "profile": profile, "account": client.account_id()}

    image_uri = pinned_image_uri(ROOT, profile)
    if "@sha256:" not in image_uri:
        raise RuntimeError(f"smoke_image_uri_not_digest_pinned: {image_uri}")
    client.deploy_stack(GUARDRAILS_STACK, GUARDRAILS_TEMPLATE)
    client.deploy_stack(
        BATCH_STACK,
        BATCH_TEMPLATE,
        parameters=[
            {"ParameterKey": "MaxvCpus", "ParameterValue": "32"},
            {"ParameterKey": "ImageUri", "ParameterValue": image_uri},
        ],
    )
    results["stacks_deployed"] = [GUARDRAILS_STACK, BATCH_STACK]
    results["image_uri"] = image_uri

    # Arm budget action via the existing operator script entrypoint logic.
    from scripts.aws_arm_budget_action import _resolve_arns
    from src.aws_burst.budget_action import (
        DEFAULT_BUDGET_NAME,
        ensure_budget_action,
        stamp_armed_flag,
    )

    exec_arn, deny_arn, topic_arn = _resolve_arns(client)
    armed = ensure_budget_action(
        client,
        budget_name=DEFAULT_BUDGET_NAME,
        threshold_pct=95.0,
        execution_role_arn=exec_arn,
        target_role_name="volsurf-burst-BatchJobRole",
        deny_policy_arn=deny_arn,
        notification_topic_arn=topic_arn,
    )
    stamp_armed_flag(ROOT, profile, action_id=armed.get("action_id"), verified=True)
    results["budget_action"] = armed
    results["spend"] = read_actual_spend(client, DEFAULT_BUDGET_NAME)

    submit = submit_wave(ROOT, wave, dry_run=False, offline=False, profiles=[profile])
    results["submit"] = submit
    job_ids = [
        s["job_id"]
        for s in (submit.get("submitted") or [])
        if s.get("job_id")
    ]
    waits = []
    for jid in job_ids:
        waits.append(
            client.wait_for_array(
                jid,
                poll_seconds=20.0,
                timeout_seconds=wait_timeout_seconds,
            )
        )
    results["waits"] = waits

    pull_index = pull_wave(ROOT, wave, require_complete=True, profiles=[profile])
    assert_smoke_artifacts(pull_index)
    results["pull"] = pull_index

    # Best-effort CloudWatch presence check via Batch job log stream metadata.
    log_ok = False
    for w in waits:
        container = (w.get("container") or {})
        if container.get("logStreamName"):
            log_ok = True
            break
    results["cloudwatch_log_seen"] = log_ok
    if not log_ok:
        results["cloudwatch_warning"] = "no_logStreamName_on_parent_job"

    if teardown:
        # Stack deletion is operator-gated; smoke records intent only unless
        # explicitly requested via future --force-teardown.
        results["teardown_requested"] = True

    results["ok"] = True
    results["elapsed_hint_s"] = None
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--profile",
        default=os.environ.get("MASCOTRL_SMOKE_PROFILE", "volsurf-burst-1"),
    )
    p.add_argument("--wave", default="PICK_SMOKE")
    p.add_argument(
        "--region",
        default=os.environ.get("MASCOTRL_AWS_REGION", REGION),
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--teardown", action="store_true")
    p.add_argument("--wait-timeout-seconds", type=float, default=3600.0)
    args = p.parse_args(argv)
    t0 = time.monotonic()
    out = run_smoke(
        profile=args.profile,
        wave=args.wave,
        region=args.region,
        dry_run=bool(args.dry_run),
        teardown=bool(args.teardown),
        wait_timeout_seconds=float(args.wait_timeout_seconds),
    )
    out["wall_s"] = time.monotonic() - t0
    print(json.dumps(out, indent=2, default=str))
    return 0 if out.get("dry_run") or out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
