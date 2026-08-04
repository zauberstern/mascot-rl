#!/usr/bin/env bash
# AWS-4: build and push image to ECR per account; record digests.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/deploy/aws_burst/scripts/_common.sh"
TAG="${1:-latest}"
REPO="${MASCOTRL_ECR_REPO:-volsurf-burst}"
mkdir -p "$ROOT/deploy/aws_burst/config"
LOCAL_TAG="volsurf-burst:local"
echo "docker build -f deploy/aws_burst/docker/Dockerfile -t $LOCAL_TAG $ROOT"
docker build -f "$ROOT/deploy/aws_burst/docker/Dockerfile" -t "$LOCAL_TAG" "$ROOT"
for p in "${BURST_PROFILES[@]}"; do
  echo "== ECR push on $p =="
  ACCOUNT="$(aws --profile "$p" --region "$REGION" sts get-caller-identity --query Account --output text)"
  URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}"
  aws --profile "$p" --region "$REGION" ecr describe-repositories --repository-names "$REPO" \
    || aws --profile "$p" --region "$REGION" ecr create-repository --repository-name "$REPO"
  aws --profile "$p" --region "$REGION" ecr get-login-password \
    | docker login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
  docker tag "$LOCAL_TAG" "${URI}:${TAG}"
  docker push "${URI}:${TAG}"
  DIGEST="$(aws --profile "$p" --region "$REGION" ecr describe-images \
    --repository-name "$REPO" --image-ids imageTag="$TAG" \
    --query 'imageDetails[0].imageDigest' --output text)"
  echo "{\"profile\":\"$p\",\"image\":\"${URI}:${TAG}\",\"digest\":\"$DIGEST\"}" \
    > "$ROOT/deploy/aws_burst/config/image_digest_${p}.json"
  echo "wrote image_digest_${p}.json digest=$DIGEST"
  # Pin Batch job definition to the digest just pushed (CFN ImageUri is
  # sticky; without this, submit keeps launching the prior digest).
  PINNED="${URI}@${DIGEST}"
  for JD in volsurf-burst-cell volsurf-burst-cell-himem volsurf-burst-cell-himem56; do
    CUR="$(aws --profile "$p" --region "$REGION" batch describe-job-definitions \
      --job-definition-name "$JD" --status ACTIVE \
      --query 'sort_by(jobDefinitions,&to_number(revision))[-1]' --output json 2>/dev/null || echo null)"
    if [[ -z "$CUR" || "$CUR" == "null" ]]; then
      continue
    fi
    python3 - "$CUR" "$PINNED" "$p" "$REGION" "$JD" <<'PY'
import json, sys, subprocess
cur = json.loads(sys.argv[1])
image, profile, region, jd_name = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
props = dict(cur["containerProperties"])
props["image"] = image
reg = {
    "jobDefinitionName": jd_name,
    "type": cur["type"],
    "containerProperties": props,
}
for k in ("retryStrategy", "timeout", "platformCapabilities"):
    if cur.get(k) is not None:
        reg[k] = cur[k]
path = f"/tmp/jd_{profile}_{jd_name}.json"
open(path, "w").write(json.dumps(reg))
subprocess.check_call([
    "aws", "--profile", profile, "--region", region,
    "batch", "register-job-definition", "--cli-input-json", f"file://{path}",
])
print(f"registered {jd_name} with {image}")
PY
  done
done
