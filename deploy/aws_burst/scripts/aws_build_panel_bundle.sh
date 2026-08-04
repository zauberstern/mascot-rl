#!/usr/bin/env bash
# Build content-hashed panel bundle (Arctic + feature-cube lake files) for Burst.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/deploy/aws_burst/scripts/_common.sh"
OUT="${1:-$ROOT/logs/aws_burst_panel_bundle}"
mkdir -p "$OUT"
python3 "$ROOT/deploy/aws_burst/scripts/build_panel_bundle.py" --out "$OUT"
echo "Upload via scripts/aws_submit_wave.py (content-addressed) or:"
for p in "${BURST_PROFILES[@]}"; do
  ACCOUNT="$(aws --profile "$p" --region "$REGION" sts get-caller-identity --query Account --output text)"
  bucket="volsurf-burst-${ACCOUNT}-panels"
  echo "  aws --profile $p s3 cp $OUT/panel_bundle.tar s3://$bucket/"
  echo "  aws --profile $p s3 cp $OUT/panel_bundle.sha256 s3://$bucket/"
  echo "  aws --profile $p s3 cp $OUT/panel_manifest.json s3://$bucket/"
done
