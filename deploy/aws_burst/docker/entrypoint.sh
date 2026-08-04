#!/usr/bin/env bash
# Container entrypoint: verify panel bundle hash, then run cell_runner.
set -euo pipefail
BUNDLE="${MASCOTRL_PANEL_BUNDLE:-/data/panel_bundle.tar}"
EXPECTED="${MASCOTRL_PANEL_SHA256:-}"
if [[ -n "$EXPECTED" && -f "$BUNDLE" ]]; then
  got="$(sha256sum "$BUNDLE" | awk '{print $1}')"
  if [[ "$got" != "$EXPECTED" ]]; then
    echo "panel_bundle_hash_mismatch: got=$got expected=$EXPECTED" >&2
    exit 2
  fi
fi
export MASCOTRL_COMPUTE_HOST="${MASCOTRL_COMPUTE_HOST:-remote}"
export MASCOTRL_REQUIREMENTS_LOCK="${MASCOTRL_REQUIREMENTS_LOCK:-/app/requirements.lock}"
# Batch reserves AWS_BATCH_* on non-array jobs; size-1 smoke passes MASCOTRL_ARRAY_INDEX.
export AWS_BATCH_JOB_ARRAY_INDEX="${AWS_BATCH_JOB_ARRAY_INDEX:-${MASCOTRL_ARRAY_INDEX:-0}}"
if [[ -z "${MASCOTRL_CONTAINER_DIGEST:-}" && -f /app/REQUIREMENTS_LOCK_SHA256 ]]; then
  export MASCOTRL_REQUIREMENTS_LOCK_SHA256="$(cat /app/REQUIREMENTS_LOCK_SHA256)"
fi
exec "$@"
