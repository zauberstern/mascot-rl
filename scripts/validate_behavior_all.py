#!/usr/bin/env python3
"""Validate behaviour metrics for ALL VAL cells (hard gate)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "logs/artifacts/spectrum/cherrypick_val"


def validate_behavior(beh_path: Path) -> dict:
    """Validate a single behaviour file."""
    stem = beh_path.stem.replace("_policy_behavior", "")
    beh = json.loads(beh_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    b = beh.get("behaviour") or beh.get("behavior") or beh
    if not isinstance(b, dict):
        return {"ok": False, "stem": stem, "errors": ["behaviour not a dict"]}

    # Missing behaviour is allowed for HAPPO dispatch_only / unavailable export.
    if beh.get("behaviour_export") == "unavailable" or not b:
        export = str(beh.get("behaviour_export") or "")
        if export == "unavailable" or "happo" in stem:
            return {"ok": True, "stem": stem, "errors": [], "note": "unavailable_ok"}

    hhi = b.get("hhi_mean")
    if hhi is None:
        errors.append("missing hhi_mean")
    elif not (0.0 <= float(hhi) <= 1.0 + 1e-6):
        errors.append(f"hhi_mean={hhi} not in [0,1]")

    turnover = b.get("turnover_mean")
    if turnover is None:
        errors.append("missing turnover_mean")
    elif float(turnover) < 0:
        errors.append(f"negative turnover_mean={turnover}")
    elif float(turnover) > 1.0:
        errors.append(f"extreme turnover_mean={turnover}")

    n_eff = b.get("n_eff_mean")
    if n_eff is None:
        errors.append("missing n_eff_mean")
    elif float(n_eff) < 1.0 - 1e-6:
        errors.append(f"n_eff_mean={n_eff} < 1")

    hold = b.get("holding_period_days")
    if hold is not None and float(hold) <= 0:
        errors.append(f"holding_period_days={hold} <= 0")

    return {"ok": len(errors) == 0, "stem": stem, "errors": errors}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--min-files", type=int, default=80)
    args = p.parse_args(argv)

    beh_files = sorted(args.dir.glob("*_policy_behavior.json"))
    results = []
    failures = []
    for beh in beh_files:
        res = validate_behavior(beh)
        results.append(res)
        if not res["ok"]:
            failures.append(res)

    print(f"Behavior files: {len(results)} (min={args.min_files})")
    print(f"Passed: {len(results) - len(failures)}")
    print(f"Failed: {len(failures)}")

    if len(results) < args.min_files:
        print(f"FAIL: only {len(results)} behaviour files")
        return 1
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  {f['stem']}: {f['errors']}")
        return 1

    print("\n=== BEHAVIOR VALIDATION PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
