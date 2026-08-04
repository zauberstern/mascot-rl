#!/usr/bin/env bash
# AWS-1: request Spot vCPU quota increase (L-34B43A08) to 64 in eu-central-1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/deploy/aws_burst/scripts/_common.sh"
rc=0
QUOTA_PROFILES=("${BURST_PROFILES[@]}")
if [[ -n "${MASCOTRL_QUOTA_PROFILES:-}" ]]; then
  IFS=',' read -ra QUOTA_PROFILES <<< "$MASCOTRL_QUOTA_PROFILES"
fi
for p in "${QUOTA_PROFILES[@]}"; do
  echo "== request quota on $p =="
  set +e
  out="$(aws --profile "$p" --region "$REGION" service-quotas request-service-quota-increase \
    --service-code ec2 \
    --quota-code "$SPOT_QUOTA_CODE" \
    --desired-value "$SPOT_QUOTA_REQUEST" 2>&1)"
  status=$?
  set -e
  if [[ $status -eq 0 ]]; then
    echo "$out"
    continue
  fi
  # Idempotent: already pending / already at value is OK; anything else fails.
  if echo "$out" | grep -Eqi 'ResourceAlreadyExists|already.*(pending|exist)|QuotaExceeded|IllegalArgumentException'; then
    echo "request may already be pending on $p"
    echo "$out"
    continue
  fi
  echo "quota_request_failed: profile=$p status=$status" >&2
  echo "$out" >&2
  rc=1
done
if [[ $rc -ne 0 ]]; then
  exit "$rc"
fi
echo "Poll with: deploy/aws_burst/scripts/aws_poll_quotas.sh"
