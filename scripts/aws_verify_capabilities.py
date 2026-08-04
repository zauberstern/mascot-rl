#!/usr/bin/env python3
"""Read-only AWS capability battery; writes account_capabilities.json."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aws_burst.aws_client import BurstClient
from src.aws_burst.profiles import (
    BURST_PROFILES,
    MAX_VCPUS_PER_ACCOUNT,
    REGION,
    SPOT_VCPU_QUOTA_CODE,
)


def _quota(client: BurstClient, quota_code: str) -> float | None:
    import boto3

    sq = boto3.Session(
        profile_name=client.profile, region_name=client.region
    ).client("service-quotas")
    try:
        resp = sq.get_service_quota(ServiceCode="ec2", QuotaCode=quota_code)
        return float(resp["Quota"]["Value"])
    except Exception:
        return None


def probe_profile(profile: str, region: str) -> dict:
    client = BurstClient(profile, region)
    import boto3

    session = boto3.Session(profile_name=profile, region_name=region)
    batch = session.client("batch")
    s3 = session.client("s3")
    cfn = session.client("cloudformation")
    budgets = session.client("budgets")
    ecr = session.client("ecr")
    identity = session.client("sts").get_caller_identity()
    account_id = str(identity["Account"])
    return {
        "profile": profile,
        "account_id": account_id,
        "arn": str(identity["Arn"]),
        "spot_vcpu_quota": _quota(client, SPOT_VCPU_QUOTA_CODE),
        "ondemand_vcpu_quota": _quota(client, "L-1216C47A"),
        "max_vcpus_configured": MAX_VCPUS_PER_ACCOUNT,
        "compute_environments": [
            ce.get("computeEnvironmentName")
            for ce in batch.describe_compute_environments().get("computeEnvironments", [])
        ],
        "job_queues": [
            jq.get("jobQueueName")
            for jq in batch.describe_job_queues().get("jobQueues", [])
        ],
        "s3_buckets": [b["Name"] for b in s3.list_buckets().get("Buckets", [])],
        "cfn_stacks": [
            s["StackName"]
            for s in cfn.list_stacks(
                StackStatusFilter=["CREATE_COMPLETE", "UPDATE_COMPLETE"]
            ).get("StackSummaries", [])
        ],
        "budgets": [
            b["BudgetName"]
            for b in budgets.describe_budgets(AccountId=account_id).get("Budgets", [])
        ],
        "ecr_repos": [
            r["repositoryName"]
            for r in ecr.describe_repositories().get("repositories", [])
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "deploy/aws_burst/config/account_capabilities.json",
    )
    args = p.parse_args(argv)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "region": REGION,
        "accounts": [probe_profile(p["profile"], REGION) for p in BURST_PROFILES],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
