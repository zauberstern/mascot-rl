#!/usr/bin/env bash
# Arm a Budget Action that denies Batch/EC2 spend above 95% of the $180 budget.
# Creates/verifies the AWS Budget Action then stamps local budget_armed_*.json.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/deploy/aws_burst/scripts/_common.sh"
mkdir -p "$ROOT/deploy/aws_burst/config"
python "$ROOT/scripts/aws_arm_budget_action.py"
echo "Budget Actions verified in AWS and local armed flags stamped."
