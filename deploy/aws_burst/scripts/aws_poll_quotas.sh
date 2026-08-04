#!/usr/bin/env bash
# Poll Spot vCPU quota approvals.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/deploy/aws_burst/scripts/_common.sh"
rc=0
for p in "${BURST_PROFILES[@]}"; do
  echo "== $p =="
  if ! aws --profile "$p" --region "$REGION" service-quotas get-service-quota \
    --service-code ec2 --quota-code "$SPOT_QUOTA_CODE" \
    --query '{QuotaName:Quota.QuotaName,Value:Quota.Value}' --output table; then
    echo "quota_poll_failed: profile=$p" >&2
    rc=1
  fi
done
exit "$rc"
