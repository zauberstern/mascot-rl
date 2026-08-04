#!/usr/bin/env bash
# K=40 three-seed bakeoff for universe-arm comparison (Rung 2).
# Arms: dyn_hrp, dyn_liquidity, dyn_crucible (CRUCIBLE-era default).
# Each arm writes to its own out-dir so the headline seal is never clobbered.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1

NPROC="$(nproc 2>/dev/null || echo 4)"
# Per-worker thread budget when seed fan-out is on (workers * threads <= 12).
SEED_WORKERS="${MASCOTRL_SEED_WORKERS:-3}"
THREADS_PER="${MASCOTRL_THREADS_PER_WORKER:-4}"
OMP_DEFAULT="$THREADS_PER"
export OMP_NUM_THREADS="${MASCOTRL_OMP_THREADS:-$OMP_DEFAULT}"
export OPENBLAS_NUM_THREADS="${MASCOTRL_OMP_THREADS:-$OMP_DEFAULT}"
export MKL_NUM_THREADS="${MASCOTRL_OMP_THREADS:-$OMP_DEFAULT}"
export NUMEXPR_NUM_THREADS="${MASCOTRL_OMP_THREADS:-$OMP_DEFAULT}"
export TORCH_NUM_THREADS="${MASCOTRL_OMP_THREADS:-$OMP_DEFAULT}"
export MASCOTRL_DUCKDB_MAX_MEMORY="${MASCOTRL_DUCKDB_MAX_MEMORY:-8GB}"
export MASCOTRL_DUCKDB_THREADS="${MASCOTRL_DUCKDB_THREADS:-4}"
export MASCOTRL_SEED_WORKERS="$SEED_WORKERS"
export MASCOTRL_THREADS_PER_WORKER="$THREADS_PER"

# Shared surface cache across bakeoff arms (fingerprint-keyed).
export MASCOTRL_SURFACE_CACHE_DIR="${MASCOTRL_SURFACE_CACHE_DIR:-$ROOT/logs/artifacts/eq_alloc/bakeoff/_shared_surface}"
mkdir -p "$MASCOTRL_SURFACE_CACHE_DIR"

# Full-path RL preflight before bakeoff arms (SKIP_PREFLIGHT=1 to bypass).
bash "$ROOT/scripts/preflight_campaign.sh"

PY="${ROOT}/.venv/bin/python"
CONFIG_DEFAULT="${MASCOTRL_CONFIG:-config/workflows/arm_equity.yaml}"
CONFIG_CRUCIBLE="${MASCOTRL_CRUCIBLE_CONFIG:-config/workflows/eq_alloc_crucible_k100.yaml}"
SEEDS="${MASCOTRL_SEEDS:-0,1,2}"
K="${MASCOTRL_BAKEOFF_K:-40}"
ARMS="${MASCOTRL_BAKEOFF_ARMS:-dyn_hrp,dyn_liquidity,dyn_crucible}"
MAX_POOL="${MASCOTRL_BAKEOFF_MAX_POOL:-200}"
DII_EPOCHS="${MASCOTRL_BAKEOFF_DII_EPOCHS:-10}"
TRAIN_STEPS="${MASCOTRL_BAKEOFF_TRAIN_STEPS:-20000}"
TRAIN_EPOCHS="${MASCOTRL_BAKEOFF_TRAIN_EPOCHS:-4}"

BAKE_ROOT="logs/artifacts/eq_alloc/bakeoff"
mkdir -p "$BAKE_ROOT"

LOCKFILE="$BAKE_ROOT/bakeoff.pid"
if [[ -f "$LOCKFILE" ]]; then
  OLD_PID="$(cat "$LOCKFILE" 2>/dev/null || true)"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[bakeoff] refusing to start: another bakeoff is running (pid=$OLD_PID)" >&2
    exit 1
  fi
  echo "[bakeoff] stale lockfile (pid=$OLD_PID); reclaiming"
fi
echo "$$" > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

TIMING="$BAKE_ROOT/rung2_timing.jsonl"
SUMMARY="$BAKE_ROOT/rung2_summary.json"
echo "{\"arms\": [], \"k\": ${K}, \"seeds\": \"${SEEDS}\", \"seed_workers\": ${SEED_WORKERS}, \"status\": \"running\"}" > "$SUMMARY"

ts() { date -Is; }

echo "[bakeoff] nproc=${NPROC} seed_workers=${SEED_WORKERS} threads_per=${THREADS_PER} duckdb_threads=${MASCOTRL_DUCKDB_THREADS}"
echo "[bakeoff] arms=${ARMS} k=${K} seeds=${SEEDS}"

IFS=',' read -ra ARM_LIST <<< "$ARMS"
for ARM in "${ARM_LIST[@]}"; do
  ARM="$(echo "$ARM" | xargs)"
  OUT_DIR="${BAKE_ROOT}/${ARM}"
  mkdir -p "$OUT_DIR"
  CONFIG="$CONFIG_DEFAULT"
  if [[ "$ARM" == "dyn_crucible" ]]; then
    CONFIG="$CONFIG_CRUCIBLE"
  fi
  echo "=== bakeoff arm=${ARM} config=${CONFIG} k=${K} seeds=${SEEDS} workers=${SEED_WORKERS} out=${OUT_DIR} $(ts) ==="
  START_EPOCH="$(date +%s)"
  set +e
  "$PY" scripts/run_eq_alloc_campaign.py \
    --config "$CONFIG" \
    --k "$K" \
    --max-pool "$MAX_POOL" \
    --dii-epochs "$DII_EPOCHS" \
    --seeds "$SEEDS" \
    --seed-workers "$SEED_WORKERS" \
    --universe-arm "$ARM" \
    --train-env-steps "$TRAIN_STEPS" \
    --train-epochs "$TRAIN_EPOCHS" \
    --min-optimizer-steps-total 40 \
    --out-dir "$OUT_DIR" \
    2>&1 | tee "${BAKE_ROOT}/${ARM}.log"
  RC=${PIPESTATUS[0]}
  set -e
  END_EPOCH="$(date +%s)"
  ELAPSED=$(( END_EPOCH - START_EPOCH ))
  echo "{\"ts\": \"$(ts)\", \"arm\": \"${ARM}\", \"config\": \"${CONFIG}\", \"k\": ${K}, \"seeds\": \"${SEEDS}\", \"seed_workers\": ${SEED_WORKERS}, \"elapsed_s\": ${ELAPSED}, \"exit\": ${RC}}" \
    >> "$TIMING"
  echo "=== arm=${ARM} done exit=${RC} elapsed_s=${ELAPSED} $(ts) ==="
  if [[ ${RC} -ne 0 ]]; then
    echo "Bakeoff arm ${ARM} failed; continuing to next arm (resume-safe)." >&2
  fi
done

"$PY" - <<'PY'
import json
from pathlib import Path
root = Path("logs/artifacts/eq_alloc/bakeoff")
arms = []
for p in sorted(root.glob("*/cpcv_path_summary.json")):
    if p.parent.name.startswith("_"):
        continue
    arms.append({"arm": p.parent.name, "summary": str(p)})
out = {"status": "finished", "arms": arms}
(root / "rung2_summary.json").write_text(json.dumps(out, indent=2) + "\n")
print("wrote", root / "rung2_summary.json", "n_arms=", len(arms))
PY

echo "Bakeoff finished. Inspect ${BAKE_ROOT}/*/cpcv_path_summary.json and ${TIMING}"
echo "Next (not this pass): apply_prereg_bakeoff_decision.py then K=100 full campaign."
