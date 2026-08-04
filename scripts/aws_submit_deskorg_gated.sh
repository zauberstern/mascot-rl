#!/usr/bin/env bash
# Refuse DESKORG until every prior thesis wave index is complete.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
import json
import sys
from pathlib import Path
from src.aws_burst.waves import WAVES

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
            return ok, (
                f"{path}: complete={data.get('complete')} "
                f"n_found={data.get('n_found')} n_errors={data.get('n_errors')}"
            )
        ok = bool(data.get("complete")) and int(data.get("n_accepted") or 0) >= int(
            data.get("n_expected") or 0
        )
        return ok, (
            f"{path}: complete={data.get('complete')} "
            f"accepted={data.get('n_accepted')}/{data.get('n_expected')}"
        )
    return False, f"no_index_for_{wave_name}"

required = ["PICK_SMOKE", "PICK", "PICK2", "K200", "FEATNET", "HYBRID"]
for w in required:
    ok, msg = index_complete(w)
    print(f"gate {w}: {msg}")
    if not ok:
        raise SystemExit(f"refuse_DESKORG_submit: prerequisite {w} incomplete")

print("GATE_OK: may submit DESKORG")
PY

echo "Submitting DESKORG ..."
.venv/bin/python scripts/aws_submit_wave.py DESKORG
