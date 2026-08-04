#!/usr/bin/env bash
# Verify sts get-caller-identity for each burst profile.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/deploy/aws_burst/scripts/_common.sh"
for p in "${BURST_PROFILES[@]}"; do
  echo "== $p =="
  aws --profile "$p" --region "$REGION" sts get-caller-identity
done
