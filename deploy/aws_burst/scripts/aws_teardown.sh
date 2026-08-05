#!/usr/bin/env bash
# AWS-10: set maxvCpus=0, delete compute stacks, keep artifacts, print cost report.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/deploy/aws_burst/scripts/_common.sh"
for p in "${BURST_PROFILES[@]}"; do
  echo "== teardown compute on $p =="
  DIGEST_FILE="$ROOT/deploy/aws_burst/config/image_digest_${p}.json"
  if [[ ! -f "$DIGEST_FILE" ]]; then
    echo "missing digest pin: $DIGEST_FILE (required so CFN does not unpin)" >&2
    exit 1
  fi
  URI="$(python3 -c "
import sys
sys.path.insert(0, '$ROOT')
from mascotrl.aws_burst.image_digest import pinned_image_uri
print(pinned_image_uri('$ROOT', '$p'))
")"
  if [[ "$URI" != *"@sha256:"* ]]; then
    echo "ImageUri must be digest-pinned (@sha256:), got: $URI" >&2
    exit 1
  fi
  if ! aws --profile "$p" --region "$REGION" cloudformation deploy \
    --stack-name volsurf-burst-batch \
    --template-file "$ROOT/deploy/aws_burst/cloudformation/10_batch_spot.yaml" \
    --parameter-overrides "MaxvCpus=0 ImageUri=$URI" \
    --capabilities CAPABILITY_NAMED_IAM; then
    echo "teardown_cfn_deploy_failed: profile=$p" >&2
    exit 1
  fi
  if ! aws --profile "$p" --region "$REGION" cloudformation delete-stack \
    --stack-name volsurf-burst-batch; then
    echo "teardown_delete_stack_failed: profile=$p" >&2
    exit 1
  fi
done
python3 - <<'PY' "$ROOT"
import json
from pathlib import Path
root = Path(__import__("sys").argv[1])
frontier = root / "deploy/aws_burst/config/cost_frontier.json"
report = {
    "artifacts_kept": str(root / "logs/artifacts/spectrum/fullgrid"),
    "frontier": json.loads(frontier.read_text()) if frontier.is_file() else {},
    "note": "Pull Cost Explorer / Budgets actuals to finalize spend per account",
}
out = root / "deploy/aws_burst/config/final_cost_report.json"
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
