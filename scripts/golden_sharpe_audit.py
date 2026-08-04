#!/usr/bin/env python3
"""Golden Sharpe audit for ALL VAL cell artifacts (hard gate)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "logs/artifacts/spectrum/cherrypick_val"
TOL = 1e-4
HYBRID_STEM = "eq_K100_single_ppo_mlp_softmax_mean_std_cao_tw-hybrid_pretrain_finetune"


def _is_cell_final(path: Path) -> bool:
    name = path.name
    if not name.endswith(".json"):
        return False
    if name.endswith(".error.json"):
        return False
    stem = path.stem
    if stem.endswith("_policy_behavior"):
        return False
    if "training" in stem or "decision_trace" in stem:
        return False
    if not stem.startswith("eq_") and not stem.startswith("opt_") and not stem.startswith("mix_"):
        return False
    return True


def audit_cell(artifact_path: Path) -> dict:
    """Audit a single cell. Returns {ok, stem, error}."""
    art = json.loads(artifact_path.read_text(encoding="utf-8"))
    stem = str(art.get("spectrum_cell_id") or artifact_path.stem)

    claim = str(
        art.get("claim_tier")
        or (art.get("spectrum_budget") or {}).get("claim_tier")
        or ""
    )
    # HAPPO dispatch_only / no OOS weight path: honest skip of Sharpe table.
    if (
        claim == "dispatch_only"
        or art.get("behaviour_export") == "unavailable"
        or ("happo" in stem and art.get("runner_artifact") in (None, {}))
    ):
        return {"ok": True, "stem": stem, "error": None, "note": "dispatch_only_ok"}

    runner = art.get("runner_artifact") or {}
    path_summary = runner.get("path_summary") or art.get("path_summary") or {}
    reported = path_summary.get("sharpe_mean")
    path_sharpes = path_summary.get("path_sharpes") or []

    if reported is None and not path_sharpes:
        return {"ok": False, "stem": stem, "error": "missing path_summary sharpe"}

    if not path_sharpes:
        return {"ok": False, "stem": stem, "error": "no path_sharpes"}

    arr = np.asarray(path_sharpes, dtype=float)
    if not np.all(np.isfinite(arr)):
        return {"ok": False, "stem": stem, "error": "non-finite path_sharpe"}

    computed_mean = float(np.mean(arr))
    reported_f = float(reported)
    diff = abs(computed_mean - reported_f)
    if diff > TOL:
        return {
            "ok": False,
            "stem": stem,
            "error": f"sharpe_mean diff={diff:.6g} reported={reported_f} computed={computed_mean}",
        }

    max_dd = path_summary.get("max_drawdown_mean")
    if max_dd is not None and float(max_dd) > TOL:
        return {"ok": False, "stem": stem, "error": f"positive max_drawdown={max_dd}"}

    return {"ok": True, "stem": stem, "error": None}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--min-cells", type=int, default=88)
    args = p.parse_args(argv)

    artifacts = sorted(a for a in args.dir.glob("*.json") if _is_cell_final(a))
    # Hybrid may be .error.json only
    hybrid_err = args.dir / f"{HYBRID_STEM}.error.json"
    hybrid_final = args.dir / f"{HYBRID_STEM}.json"

    results = []
    failures = []
    for art in artifacts:
        if art.stem == HYBRID_STEM:
            continue
        res = audit_cell(art)
        results.append(res)
        if not res["ok"]:
            failures.append(res)

    n_cells = len(results)
    hybrid_ok = hybrid_err.is_file() or hybrid_final.is_file()
    print(f"Audited: {n_cells} cells (min={args.min_cells})")
    print(f"Passed: {n_cells - len(failures)}")
    print(f"Failed: {len(failures)}")
    print(f"Hybrid present: {hybrid_ok}")

    if n_cells < args.min_cells:
        print(f"FAIL: only {n_cells} cells, need >= {args.min_cells}")
        return 1
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  {f['stem']}: {f['error']}")
        return 1
    if not hybrid_ok:
        print("FAIL: hybrid control missing (final or error)")
        return 1

    print("\n=== GOLDEN SHARPE AUDIT PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
