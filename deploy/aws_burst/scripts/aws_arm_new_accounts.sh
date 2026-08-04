#!/usr/bin/env bash
# Arm volsurf-burst-4..6 after operator fills .env + ~/.aws profiles.
# Usage: bash deploy/aws_burst/scripts/aws_arm_new_accounts.sh 4 5 6
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile_suffix> ...  e.g. 4 5 6" >&2
  exit 1
fi

for n in "$@"; do
  profile="volsurf-burst-${n}"
  echo "=== ${profile} ==="
  if ! aws --profile "$profile" --region eu-central-1 sts get-caller-identity >/dev/null 2>&1; then
    echo "FAIL: profile ${profile} not configured in ~/.aws" >&2
    exit 2
  fi
  bash deploy/aws_burst/scripts/aws_request_quotas.sh "$profile"
  bash deploy/aws_burst/scripts/aws_deploy_guardrails.sh "$profile"
  bash deploy/aws_burst/scripts/aws_arm_budget_action.sh "$profile"
  digest_file="deploy/aws_burst/image_digest_${profile}.json"
  if [[ ! -f "$digest_file" ]]; then
    cp deploy/aws_burst/image_digest_volsurf-burst-1.json "$digest_file"
    echo "replicated digest pin -> $digest_file"
  fi
  bash deploy/aws_burst/scripts/aws_build_panel_bundle.sh "$profile"
  .venv/bin/python scripts/aws_submit_wave.py PICK_SMOKE --profile "$profile" --force
  echo "OK ${profile}"
done

echo "All profiles armed. Append to src/aws_burst/profiles.py BURST_PROFILES atomically with budget_armed_*.json."
