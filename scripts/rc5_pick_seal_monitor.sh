#!/usr/bin/env bash
# RC5: wait for a fresh PICK seal, then pull / codename / separation-check.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
MIN_POLL="2026-08-29T18:46:00"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

log "RC5 PICK seal monitor start (require polled_at>=${MIN_POLL}, complete, n_found>=n_expected)"

while true; do
  if [[ -f logs/aws_burst_watch_PICK.json ]]; then
    mapfile -t status < <(.venv/bin/python - <<PY
import json
d = json.load(open("logs/aws_burst_watch_PICK.json"))
polled = str(d.get("polled_at") or "")
complete = bool(d.get("complete"))
n_found = int(d.get("n_found") or 0)
n_exp = int(d.get("n_expected") or 53)
n_err = int(d.get("n_errors") or 0)
fresh = polled >= "${MIN_POLL}"
seal = complete and fresh and n_found >= n_exp and n_err == 0
print(1 if seal else 0)
print(f"complete={complete} fresh={fresh} found={n_found}/{n_exp} errors={n_err} polled={polled}")
PY
)
    seal="${status[0]}"
    log "watch_status ${status[1]}"
    if [[ "$seal" == "1" ]]; then
      log "PICK sealed (fresh); pulling artifacts"
      .venv/bin/python scripts/aws_pull_artifacts.py --wave PICK --allow-digest-drift \
        2>&1 | tee logs/aws_burst/rc5_pull_PICK.log | tail -80
      log "assign_behavior_codenames"
      .venv/bin/python scripts/assign_behavior_codenames.py \
        --dir logs/artifacts/spectrum/cherrypick \
        2>&1 | tee logs/aws_burst/rc5_codenames_PICK.log | tail -60
      log "separation validation"
      .venv/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path("logs/artifacts/spectrum/cherrypick")
cells = []
for p in sorted(root.glob("*.json")):
    if p.name == "index.json" or p.name.endswith(".error.json"):
        continue
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    if not isinstance(d, dict):
        continue
    pb = d.get("policy_behavior") or {}
    cells.append(
        {
            "stem": p.stem,
            "l1": float(pb.get("l1_vs_ew", pb.get("L1_vs_EW", float("nan")))),
            "turnover": float(pb.get("turnover_mean", float("nan"))),
            "max_w": float(pb.get("max_w", float("nan"))),
            "ew_collapse": bool(pb.get("equal_weight_collapse_detected", False)),
            "archetype": str(
                pb.get("archetype_primary") or d.get("archetype_primary") or ""
            ),
            "head": str(
                (d.get("cfg") or {}).get("weight_head")
                or d.get("weight_head")
                or ""
            ),
        }
    )

n = len(cells)
ew_rate = float(np.mean([c["ew_collapse"] for c in cells])) if n else float("nan")
arch = sorted({c["archetype"] for c in cells if c["archetype"]})
softmax = [c for c in cells if "softmax" in c["head"]]
turn_ok = (
    float(np.mean([c["turnover"] > 0.005 for c in softmax])) if softmax else float("nan")
)
maxw_ok = (
    float(np.mean([c["max_w"] > 0.015 for c in softmax])) if softmax else float("nan")
)
report = {
    "n_cells": n,
    "equal_weight_collapse_rate": ew_rate,
    "collapse_ok": bool(ew_rate < 0.30) if n else False,
    "archetypes": arch,
    "archetype_ok": len(arch) >= 3,
    "softmax_turnover_gt_0.005_rate": turn_ok,
    "softmax_max_w_gt_0.015_rate": maxw_ok,
    "median_l1_vs_ew": float(np.nanmedian([c["l1"] for c in cells])) if n else None,
    "median_turnover": float(np.nanmedian([c["turnover"] for c in cells])) if n else None,
}
Path("logs/aws_burst/rc5_separation_PICK.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
print(
    "SEPARATION_OK"
    if (report["collapse_ok"] and report["archetype_ok"])
    else "SEPARATION_NEEDS_REVIEW"
)
PY
      log "PICK post-seal done"
      echo RC5_MONITOR_COMPLETE
      exit 0
    fi
  else
    log "waiting for logs/aws_burst_watch_PICK.json"
  fi
  sleep 180
done
