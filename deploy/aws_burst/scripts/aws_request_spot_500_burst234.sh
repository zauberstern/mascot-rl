#!/usr/bin/env bash
# Request 500 Spot vCPU (L-34B43A08) on burst-2..burst-4 in eu-central-1.
#
# Service Quotas guide notes:
# - Smaller increases vs current applied quota auto-approve more often (~<=2x).
# - Larger jumps open a Support case (CASE_OPENED); add use-case text in Support Center.
# - Only one open increase request is allowed per quota (ResourceAlreadyExists).
#
# After CLI submit, paste deploy/aws_burst/config/quota_increase_use_case_generic.txt
# into each Support case (Support Center web console; Basic support cannot use Support API).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/deploy/aws_burst/scripts/_common.sh"

# Fixed target for this script; do not inherit SPOT_QUOTA_REQUEST default (64) from _common.sh.
DESIRED=500
QUOTA_PROFILES=(volsurf-burst-2 volsurf-burst-3 volsurf-burst-4)
USE_CASE="$ROOT/deploy/aws_burst/config/quota_increase_use_case_generic.txt"

echo "Target: $DESIRED Spot vCPUs (${SPOT_QUOTA_CODE}) in $REGION"
echo "Profiles: ${QUOTA_PROFILES[*]}"
echo "Use-case paste file: $USE_CASE"
echo

rc=0
for p in "${QUOTA_PROFILES[@]}"; do
  echo "== $p =="
  applied="$(aws --profile "$p" --region "$REGION" service-quotas get-service-quota \
    --service-code ec2 --quota-code "$SPOT_QUOTA_CODE" \
    --query 'Quota.Value' --output text 2>/dev/null || echo "?")"
  echo "applied_quota=$applied desired=$DESIRED"

  if [[ "$applied" != "?" && "$applied" != "None" ]]; then
    # shellcheck disable=SC2086
    if awk -v a="$applied" -v d="$DESIRED" 'BEGIN { exit !(a >= d) }'; then
      echo "skip: applied quota already >= desired"
      echo
      continue
    fi
  fi

  pending_json="$(aws --profile "$p" --region "$REGION" service-quotas \
    list-requested-service-quota-change-history-by-quota \
    --service-code ec2 --quota-code "$SPOT_QUOTA_CODE" \
    --query 'RequestedQuotas[?Status==`PENDING` || Status==`CASE_OPENED` || Status==`NOT_APPROVED`]' \
    --output json 2>/dev/null || echo '[]')"
  pending_n="$(python3 -c "import json,sys; print(len(json.load(sys.stdin)))" <<<"$pending_json")"
  if [[ "$pending_n" != "0" ]]; then
    echo "blocked: open quota request already exists (only one open request per quota)."
    tmp_pending="$(mktemp)"
    printf '%s' "$pending_json" >"$tmp_pending"
    python3 - <<'PY' "$tmp_pending" "$DESIRED" "$USE_CASE"
import json, sys
from pathlib import Path
pending = json.loads(Path(sys.argv[1]).read_text())
desired = sys.argv[2]
use_case = sys.argv[3]
for row in pending:
    print(
        f"  status={row.get('Status')} desired={row.get('DesiredValue')} "
        f"caseId={row.get('CaseId')} id={row.get('Id')}"
    )
print("  Action: open AWS Support Center -> Cases -> caseId above -> reply with:")
print(f"           {use_case}")
print(f"  Ask Support to amend desired value to {desired} if it differs.")
print("  To submit a fresh CLI request instead: close that case in Support Center, then re-run this script.")
PY
    rm -f "$tmp_pending"
    echo
    rc=2
    continue
  fi

  set +e
  out="$(aws --profile "$p" --region "$REGION" service-quotas request-service-quota-increase \
    --service-code ec2 \
    --quota-code "$SPOT_QUOTA_CODE" \
    --desired-value "$DESIRED" 2>&1)"
  status=$?
  set -e
  if [[ $status -eq 0 ]]; then
    echo "$out"
    echo "  Next: if status becomes CASE_OPENED, paste $USE_CASE into the Support case."
  else
    echo "quota_request_failed: profile=$p status=$status" >&2
    echo "$out" >&2
    rc=1
  fi
  echo
done

if [[ $rc -eq 0 ]]; then
  echo "Submitted. Poll: deploy/aws_burst/scripts/aws_poll_quotas.sh"
elif [[ $rc -eq 2 ]]; then
  echo "No new CLI submits (open requests present). Use Support case correspondence above."
  exit 2
else
  exit "$rc"
fi
