#!/usr/bin/env python3
"""Create scoped IAM operator user per burst account (run once per profile)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import boto3

from mascotrl.aws_burst.profiles import BURST_PROFILES, REGION

USER_NAME = "volsurf-burst-operator"
POLICY_NAME = "volsurf-burst-operator-policy"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", required=True, choices=[x["profile"] for x in BURST_PROFILES])
    args = p.parse_args(argv)
    session = boto3.Session(profile_name=args.profile, region_name=REGION)
    iam = session.client("iam")
    policy_doc = json.loads(
        (ROOT / "deploy/aws_burst/policies/operator_policy.json").read_text(encoding="utf-8")
    )
    try:
        iam.create_user(UserName=USER_NAME)
    except iam.exceptions.EntityAlreadyExistsException:
        pass
    iam.put_user_policy(
        UserName=USER_NAME,
        PolicyName=POLICY_NAME,
        PolicyDocument=json.dumps(policy_doc),
    )
    resp = iam.create_access_key(UserName=USER_NAME)
    cred = resp["AccessKey"]
    out = ROOT / "deploy/aws_burst/config" / f"operator_credentials_{args.profile}.json"
    out.write_text(
        json.dumps(
            {
                "profile": args.profile,
                "user": USER_NAME,
                "access_key_id": cred["AccessKeyId"],
                "secret_access_key": cred["SecretAccessKey"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out} (gitignored pattern: keep out of commits)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
