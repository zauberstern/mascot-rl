#!/usr/bin/env bash
# Shared env for burst scripts.
REGION="${AWS_REGION:-eu-central-1}"
BURST_PROFILES=(
  volsurf-burst-1
  volsurf-burst-2
  volsurf-burst-3
  volsurf-burst-4
)
SPOT_QUOTA_CODE="L-34B43A08"
SPOT_QUOTA_REQUEST="${SPOT_QUOTA_REQUEST:-64}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUDGET_USD="$(python3 -c "import sys; sys.path.insert(0,'$ROOT/src'); from mascotrl.aws_burst.profiles import BUDGET_USD; print(BUDGET_USD)")"
