#!/usr/bin/env python3
"""Verify/create AWS Budgets hard-deny action and stamp local armed flags."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from botocore.config import Config

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aws_burst.aws_client import BurstClient
from src.aws_burst.budget_action import (
    DEFAULT_BUDGET_NAME,
    ensure_budget_action,
    stamp_armed_flag,
)
from src.aws_burst.profiles import BURST_PROFILES, REGION

TARGET_ROLE = "volsurf-burst-BatchJobRole"
EXEC_ROLE_NAME = "volsurf-burst-budgets-action-exec"
DENY_POLICY_NAME = "volsurf-burst-spend-deny"
_RETRY = Config(retries={"mode": "standard", "max_attempts": 5})


def _resolve_arns(client: BurstClient) -> tuple[str, str, str | None]:
    """Resolve execution role, deny policy, and optional SNS topic ARNs."""
    acct = client.account_id()
    iam = client._session.client("iam", config=_RETRY)
    exec_arn = f"arn:aws:iam::{acct}:role/{EXEC_ROLE_NAME}"
    deny_arn = f"arn:aws:iam::{acct}:policy/{DENY_POLICY_NAME}"
    iam.get_role(RoleName=EXEC_ROLE_NAME)
    iam.get_policy(PolicyArn=deny_arn)
    topic_arn = None
    try:
        sns = client._session.client("sns", region_name=REGION, config=_RETRY)
        for t in sns.list_topics().get("Topics") or []:
            arn = str(t.get("TopicArn") or "")
            if arn.endswith(":volsurf-burst-budget"):
                topic_arn = arn
                break
    except Exception:  # noqa: BLE001 - SNS optional for arm stamp
        topic_arn = None
    return exec_arn, deny_arn, topic_arn


def main() -> int:
    results = []
    for profile_info in BURST_PROFILES:
        profile = profile_info["profile"]
        client = BurstClient(profile, REGION)
        exec_arn, deny_arn, topic_arn = _resolve_arns(client)
        out = ensure_budget_action(
            client,
            budget_name=DEFAULT_BUDGET_NAME,
            threshold_pct=95.0,
            execution_role_arn=exec_arn,
            target_role_name=TARGET_ROLE,
            deny_policy_arn=deny_arn,
            notification_topic_arn=topic_arn,
        )
        path = stamp_armed_flag(
            ROOT,
            profile,
            action_id=out.get("action_id"),
            verified=True,
        )
        results.append(
            {
                "profile": profile,
                "action_id": out.get("action_id"),
                "created": out.get("created"),
                "armed_flag": str(path),
            }
        )
    print(json.dumps({"armed": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
