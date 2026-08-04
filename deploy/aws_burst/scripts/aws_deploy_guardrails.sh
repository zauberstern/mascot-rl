#!/usr/bin/env bash
# AWS-2: deploy guardrails stack per account.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/deploy/aws_burst/scripts/_common.sh"
EMAILS=(
  burst-operator@example.invalid
  burst-operator@example.invalid
  burst-operator@example.invalid
  burst-operator@example.invalid
  burst-operator@example.invalid
)
i=0
for p in "${BURST_PROFILES[@]}"; do
  email="${EMAILS[$i]}"
  echo "== guardrails $p ($email) =="
  aws --profile "$p" --region "$REGION" cloudformation deploy \
    --stack-name volsurf-burst-guardrails \
    --template-file "$ROOT/deploy/aws_burst/cloudformation/00_guardrails.yaml" \
    --parameter-overrides "NotificationEmail=$email" "BudgetAmount=$BUDGET_USD" \
    --capabilities CAPABILITY_NAMED_IAM
  i=$((i + 1))
done
echo "Confirm SNS email subscriptions, then arm Budget Action deny via console or aws_arm_budget_action.sh"
