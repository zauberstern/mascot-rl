#!/usr/bin/env bash
# Chain remaining Burst waves after PICK2: K200 → FEATNET → HYBRID → DESKORG.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$ROOT"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

wait_wave_complete() {
  local wave="$1"
  local expected="$2"
  local timeout_s="${3:-172800}"
  log "wait $wave expected=$expected"
  .venv/bin/python scripts/aws_burst_s3_watch.py "$wave" \
    --expected "$expected" \
    --out "logs/aws_burst_watch_${wave}.json" \
    --poll-seconds 90 \
    --timeout-seconds "$timeout_s"
  .venv/bin/python scripts/aws_pull_artifacts.py --wave "$wave"
  local idx
  idx="$(.venv/bin/python - <<PY
from mascotrl.aws_burst.waves import WAVES
print(WAVES["$wave"].out_subdir)
PY
)"
  python3 - <<PY
import json
from pathlib import Path
p = Path("logs/artifacts/spectrum") / "$idx" / "index.json"
assert p.is_file(), p
d = json.loads(p.read_text())
assert d.get("complete"), d
print("OK", p, "accepted", d.get("n_accepted"), "/", d.get("n_expected"))
PY
}

# --- PICK2 already submitted; wait ---
if [[ ! -f logs/artifacts/spectrum/cherrypick/narrative/index.json ]] || \
   ! python3 -c "import json;d=json.load(open('logs/artifacts/spectrum/cherrypick/narrative/index.json'));raise SystemExit(0 if d.get('complete') else 1)"; then
  wait_wave_complete PICK2 7 || true
  # Retry incomplete / OOM stems (e.g. CPPO SIGKILL) once jobs drain.
  if ! python3 -c "import json;from pathlib import Path;p=Path('logs/artifacts/spectrum/cherrypick/narrative/index.json');
import sys
sys.exit(0 if p.is_file() and json.loads(p.read_text()).get('complete') else 1)"; then
    log "PICK2 incomplete; force-resubmit remaining stems"
    .venv/bin/python scripts/aws_submit_wave.py PICK2 --force
    wait_wave_complete PICK2 7
  fi
fi
log "PICK2 green"

# --- K200 ---
if [[ ! -f logs/artifacts/spectrum/cherrypick/k200/index.json ]] || \
   ! python3 -c "import json;d=json.load(open('logs/artifacts/spectrum/cherrypick/k200/index.json'));raise SystemExit(0 if d.get('complete') and int(d.get('n_accepted') or 0)>=33 else 1)"; then
  log "submit K200"
  .venv/bin/python scripts/aws_submit_wave.py K200
  wait_wave_complete K200 33
fi
log "K200 green"

# --- FEATNET then HYBRID ---
bash scripts/aws_submit_featnet_hybrid_gated.sh FEATNET
wait_wave_complete FEATNET 53
bash scripts/aws_submit_featnet_hybrid_gated.sh HYBRID
wait_wave_complete HYBRID 3
log "FEATNET+HYBRID green"

# --- DESKORG last ---
bash scripts/aws_submit_deskorg_gated.sh
wait_wave_complete DESKORG 1

# Post: interpretability + contrast macros + thesis numbers best-effort
.venv/bin/python scripts/rehydrate_deskorg_peer.py || true
if [[ -f scripts/run_interpretability.py ]]; then
  .venv/bin/python scripts/run_interpretability.py \
    --artifact-dir logs/artifacts/spectrum/cherrypick_deskorg || true
fi
.venv/bin/python scripts/build_deskorg_contrast.py || true
.venv/bin/python scripts/assign_behavior_codenames.py --dir logs/artifacts/spectrum/cherrypick || true
.venv/bin/python scripts/build_thesis_numbers.py --strict || true
log "CHAIN_DONE"
