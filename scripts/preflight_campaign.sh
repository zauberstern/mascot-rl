#!/usr/bin/env bash
# Full-path campaign preflight (no --skip-rl) before multi-hour launches.
#
# Usage:
#   bash scripts/preflight_campaign.sh
#
# Bypass (operator-owned; not for CI greenwashing of broken late stages):
#   SKIP_PREFLIGHT=1 bash scripts/run_full_campaign.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "${SKIP_PREFLIGHT:-0}" == "1" ]]; then
  echo "[preflight] SKIP_PREFLIGHT=1 set; bypassing full-path preflight" >&2
  exit 0
fi

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "[preflight] missing $PY" >&2
  exit 2
fi

# Fail closed: lake + allowlist must be present. Soft-skip used to greenwash
# unmounted-lake launches and is no longer allowed.
LAKE="${MASCOTRL_LAKE_DIR:-/mnt/volsurf/volsurf_data_lake}"
if [[ ! -d "$LAKE" ]]; then
  echo "[preflight] FAIL: lake not mounted at ${LAKE} (set MASCOTRL_LAKE_DIR)" >&2
  exit 1
fi
ALLOW="${ROOT}/config/signal_allowlist.json"
if [[ ! -f "$ALLOW" ]]; then
  echo "[preflight] FAIL: missing ${ALLOW}" >&2
  exit 1
fi
if ! "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('allowlist') else 1)" "$ALLOW"; then
  echo "[preflight] FAIL: empty allowlist in ${ALLOW}" >&2
  exit 1
fi
export MASCOTRL_LAKE_DIR="$LAKE"
echo "[preflight] lake ok: ${LAKE}"
echo "[preflight] allowlist ok: ${ALLOW}"

echo "[preflight] running full-path RL preflight (dyn_liquidity + dyn_hrp, k=6, 2 seeds)"
exec "$PY" -m pytest \
  tests/test_campaign_full_path_preflight.py \
  -m "slow and integration" \
  -q --tb=short
