#!/usr/bin/env bash
# VAL 100% attestation gate. Exit 0 only when hard checks pass.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="$ROOT"

DIR="${1:-logs/artifacts/spectrum/cherrypick_val}"
echo "=== VAL 100% ATTESTATION GATE ==="
echo "dir=$DIR"

n_arts=$(find "$DIR" -maxdepth 1 -name '*.json' ! -name '*_policy_behavior.json' ! -name '*.error.json' | wc -l)
echo "Artifacts: $n_arts (need >= 88)"
[[ "$n_arts" -ge 88 ]] || { echo "FAIL: Missing artifacts"; exit 1; }

result=$(python scripts/validate_val_subset.py --dir "$DIR")
n_passed=$(echo "$result" | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('n_passed',0))")
ok=$(echo "$result" | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('ok', False))")
echo "Validation passed: $n_passed ok=$ok"
[[ "$n_passed" -ge 88 ]] || { echo "$result" | python -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d.get('anomalies',[])[:5], indent=2))"; echo "FAIL: Validation"; exit 1; }

python scripts/golden_sharpe_audit.py --dir "$DIR" --min-cells 88
python scripts/validate_behavior_all.py --dir "$DIR" --min-files 70

pytest tests/test_reward_parity.py tests/test_cpcv.py tests/test_pit_leakage.py tests/test_validate_val_subset.py -q

echo "=== VAL ATTESTATION GATE PASSED ==="
echo "You may now proceed to thesis experiments."
