#!/usr/bin/env bash
# Gate FEATNET / HYBRID submit until K200 (and preferred PICK/PICK2) are complete.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
WAVE="${1:-}"
if [[ -z "$WAVE" || ( "$WAVE" != "FEATNET" && "$WAVE" != "HYBRID" ) ]]; then
  echo "usage: $0 FEATNET|HYBRID" >&2
  exit 2
fi

python3 - <<'PY' "$WAVE"
import json, sys
from pathlib import Path
from mascotrl.aws_burst.waves import WAVES

wave = sys.argv[1]
root = Path(".").resolve()

def index_complete(wave_name: str) -> tuple[bool, str]:
    spec = WAVES.get(wave_name)
    candidates = []
    if spec is not None:
        candidates.append(root / "logs/artifacts/spectrum" / spec.out_subdir / "index.json")
    candidates.append(root / f"logs/aws_burst_watch_{wave_name}.json")
    for path in candidates:
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        if "polled_at" in data or path.name.startswith("aws_burst_watch"):
            ok = bool(data.get("complete")) and int(data.get("n_errors") or 0) == 0
            return ok, f"{path}: complete={data.get('complete')} n_found={data.get('n_found')} n_errors={data.get('n_errors')}"
        ok = bool(data.get("complete")) and int(data.get("n_accepted") or 0) >= int(
            data.get("n_expected") or 0
        )
        return ok, (
            f"{path}: complete={data.get('complete')} "
            f"accepted={data.get('n_accepted')}/{data.get('n_expected')}"
        )
    return False, f"no_index_for_{wave_name}"

required = ["K200"]
preferred = ["PICK", "PICK2"]
for w in required:
    ok, msg = index_complete(w)
    print(f"gate {w}: {msg}")
    if not ok:
        raise SystemExit(f"refuse_{wave}_submit: prerequisite {w} incomplete")
for w in preferred:
    ok, msg = index_complete(w)
    print(f"prefer {w}: {msg}")
    if not ok:
        print(f"warning: preferred {w} incomplete; continuing because K200 passed", flush=True)

print(f"GATE_OK: may submit {wave}")
PY

echo "Submitting $WAVE ..."
.venv/bin/python scripts/aws_submit_wave.py "$WAVE"
