#!/usr/bin/env bash
# AWS-5: deploy Batch Spot compute stack (digest-pinned ImageUri per profile).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/deploy/aws_burst/scripts/_common.sh"
MAXV="${1:-32}"
INSTANCE_TYPES="${2:-m7i-flex.large}"
JOBMEM="${3:-6912}"
HIMEM="${4:-16384}"
DEPLOY_PROFILES=("${BURST_PROFILES[@]}")
if [[ -n "${MASCOTRL_DEPLOY_PROFILES:-}" ]]; then
  IFS=',' read -ra DEPLOY_PROFILES <<< "$MASCOTRL_DEPLOY_PROFILES"
fi
for p in "${DEPLOY_PROFILES[@]}"; do
  DIGEST_FILE="$ROOT/deploy/aws_burst/config/image_digest_${p}.json"
  if [[ ! -f "$DIGEST_FILE" ]]; then
    echo "missing digest pin: $DIGEST_FILE (run aws_build_push_image.sh)" >&2
    exit 1
  fi
  URI="$(python3 -c "
import sys
sys.path.insert(0, '$ROOT')
from src.aws_burst.image_digest import pinned_image_uri
print(pinned_image_uri('$ROOT', '$p'))
")"
  if [[ "$URI" != *"@sha256:"* ]]; then
    echo "ImageUri must be digest-pinned (@sha256:), got: $URI" >&2
    exit 1
  fi
  if [[ "$URI" == *"public.ecr.aws/docker/library/python"* ]]; then
    echo "refusing public.ecr.aws/docker/library/python default image" >&2
    exit 1
  fi
  echo "== batch $p maxvCpus=$MAXV types=$INSTANCE_TYPES mem=$JOBMEM himem=$HIMEM =="
  aws --profile "$p" --region "$REGION" cloudformation deploy \
    --stack-name volsurf-burst-batch \
    --template-file "$ROOT/deploy/aws_burst/cloudformation/10_batch_spot.yaml" \
    --parameter-overrides \
      "MaxvCpus=$MAXV" \
      "ImageUri=$URI" \
      "InstanceTypes=$INSTANCE_TYPES" \
      "JobMemoryMiB=$JOBMEM" \
      "HimemJobMemoryMiB=$HIMEM" \
    --capabilities CAPABILITY_NAMED_IAM
done
