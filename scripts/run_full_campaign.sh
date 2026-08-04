#!/usr/bin/env bash
# Full-scale equity allocation campaign (headline arm): DII causal universe,
# real signal gate, HAPPO+CMDP PPO under CPCV(6,2), full seed grid.
#
# This is the multi-hour, unattended production run behind the thesis's
# \TBD placeholders (thesis/fs_thesis.tex, Appendix on Reproducibility).
# The smoke-scale structural validation (Workstream F) is a separate,
# reduced-seed / reduced-pool invocation of the same two scripts below and
# is not what this wrapper runs.
#
# Usage:
#   nohup bash scripts/run_full_campaign.sh >> logs/run_full_campaign.stdout 2>&1 &
#
# Env overrides:
#   MASCOTRL_CONFIG        workflow YAML (default: config/workflows/arm_equity.yaml)
#   MASCOTRL_SEEDS         comma-separated seeds (default: 0..9, matching the config's eval_seeds)
#   MASCOTRL_K             DII-selected universe size (default: 100)
#   MASCOTRL_GATE_MAX_POOL signal-gate candidate pool; 0 = full universe (default: 0)
#   SKIP_SIGNAL_GATE=1    reuse the existing config/signal_allowlist.json instead of
#                         re-running the gate (fails closed downstream if it is empty)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1

# Cap BLAS/Torch fan-out. Unbounded nproc fan-out multiplied DuckDB/pandas
# peak RSS and helped OOM the full campaign even after signal-gate batching.
NPROC="$(nproc 2>/dev/null || echo 4)"
OMP_DEFAULT=$(( NPROC > 4 ? 4 : NPROC ))
export OMP_NUM_THREADS="${MASCOTRL_OMP_THREADS:-$OMP_DEFAULT}"
export OPENBLAS_NUM_THREADS="${MASCOTRL_OMP_THREADS:-$OMP_DEFAULT}"
export MKL_NUM_THREADS="${MASCOTRL_OMP_THREADS:-$OMP_DEFAULT}"
export NUMEXPR_NUM_THREADS="${MASCOTRL_OMP_THREADS:-$OMP_DEFAULT}"
export TORCH_NUM_THREADS="${MASCOTRL_OMP_THREADS:-$OMP_DEFAULT}"
export OMP_WAIT_POLICY=PASSIVE
export MASCOTRL_DUCKDB_MAX_MEMORY="${MASCOTRL_DUCKDB_MAX_MEMORY:-4GB}"
export MASCOTRL_DUCKDB_THREADS="${MASCOTRL_DUCKDB_THREADS:-4}"

# Full-path RL preflight (catches nested-WFO / stats / gates / thesis schema).
# Set SKIP_PREFLIGHT=1 only for operator-owned bypass after a green preflight.
bash "$ROOT/scripts/preflight_campaign.sh"

CONFIG="${MASCOTRL_CONFIG:-config/workflows/arm_equity.yaml}"
SEEDS="${MASCOTRL_SEEDS:-0,1,2,3,4,5,6,7,8,9}"
K="${MASCOTRL_K:-100}"
GATE_MAX_POOL="${MASCOTRL_GATE_MAX_POOL:-0}"
ALLOWLIST_PATH="config/signal_allowlist.json"
PY="$ROOT/.venv/bin/python"

mkdir -p "$ROOT/logs/artifacts/eq_alloc"

LOCKFILE="$ROOT/logs/artifacts/eq_alloc/campaign.pid"
if [[ -f "$LOCKFILE" ]]; then
  OLD_PID="$(cat "$LOCKFILE" 2>/dev/null || true)"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[full] refusing to start: another campaign is already running (pid=$OLD_PID, lockfile=$LOCKFILE)" >&2
    exit 1
  fi
  echo "[full] stale lockfile found (pid=$OLD_PID not running); reclaiming $LOCKFILE"
fi
echo "$$" > "$LOCKFILE"

# On SIGTERM/SIGINT (systemd stop, Ctrl-C) only release the lockfile so a
# restarted service does not see a stale "already running" pid. Flushing
# in-progress campaign state (manifest / cpcv_seed_*.json / checkpoints) is
# the Python process's job (atomic_write_json + fold/episode checkpoints),
# not this wrapper's; the wrapper never buffers campaign data itself.
cleanup() {
  rm -f "$LOCKFILE"
}
trap 'cleanup; exit 143' SIGTERM
trap 'cleanup; exit 130' SIGINT
trap cleanup EXIT

ts() { date -Is; }

echo "========== run_full_campaign START $(ts) ==========" 
echo "[full] config=$CONFIG seeds=$SEEDS k=$K gate_max_pool=$GATE_MAX_POOL"

if [[ "${SKIP_SIGNAL_GATE:-0}" == "1" ]]; then
  echo "[full] SKIP_SIGNAL_GATE=1: reusing existing $ALLOWLIST_PATH"
else
  echo "[full] signal gate: PIT admission over the 2003-2012 selection window $(ts)"
  "$PY" scripts/run_signal_gate.py --max-pool "$GATE_MAX_POOL" --out "$ALLOWLIST_PATH"
  echo "[full] signal gate wrote $ALLOWLIST_PATH $(ts)"
fi

echo "[full] equity allocation campaign: HAPPO+CMDP PPO under CPCV(6,2) $(ts)"
CAMPAIGN_ARGS=(--config "$CONFIG" --k "$K" --seeds "$SEEDS")
# Optional override; when unset the campaign reads universe_arm from YAML.
if [[ -n "${MASCOTRL_UNIVERSE_ARM:-}" ]]; then
  CAMPAIGN_ARGS+=(--universe-arm "$MASCOTRL_UNIVERSE_ARM")
fi
# WFO alongside CPCV by default (campaign --no-wfo to skip).
if [[ "${MASCOTRL_NO_WFO:-0}" != "1" ]]; then
  : # default-on once campaign inverts the flag; keep hook for env override
fi
"$PY" scripts/run_eq_alloc_campaign.py "${CAMPAIGN_ARGS[@]}"
echo "[full] campaign finished $(ts)"

SUMMARY="$ROOT/logs/artifacts/eq_alloc/cpcv_path_summary.json"
HASH="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('run_config_hash') or '')" "$SUMMARY")"
echo "[full] populating thesis/generated/numbers.tex from campaign artifacts (strict) $(ts)"
"$PY" scripts/build_thesis_numbers.py \
  --results "$SUMMARY" \
  --out "$ROOT/thesis/generated/numbers.tex" \
  --strict \
  --require-k "$K" \
  --require-config-hash "$HASH"
echo "[full] exporting thesis figures $(ts)"
"$PY" scripts/export_thesis_figures.py \
  --artifacts "$ROOT/logs/artifacts/eq_alloc" \
  --spectrum "$ROOT/logs/artifacts/spectrum" \
  --behaviour "$ROOT/logs/artifacts/policy_behavior_panel" \
  --out "$ROOT/thesis/figures"

echo "========== run_full_campaign END $(ts) =========="
