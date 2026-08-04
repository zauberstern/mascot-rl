#!/usr/bin/env python3
"""Validate HEAD-EQ / HEAD-SURF-OFF ``cpcv_path_summary.json`` (roadmap W2 checklist)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_HEAD = ROOT / "logs" / "artifacts" / "eq_alloc" / "cpcv_path_summary.json"
DEFAULT_SURF_OFF = (
    ROOT / "logs" / "artifacts" / "eq_alloc" / "ablation_surface_off" / "cpcv_path_summary.json"
)


def _finite(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def validate_summary(
    path: Path,
    *,
    label: str,
    require_k: int = 100,
    require_n_seeds: int = 10,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not path.is_file():
        return {"ok": False, "label": label, "path": str(path), "errors": ["missing_file"]}
    try:
        results = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "label": label, "path": str(path), "errors": [f"unreadable: {exc}"]}

    k = results.get("k")
    if int(k or 0) != int(require_k):
        errors.append(f"k_mismatch: got {k!r} want {require_k}")

    conf = results.get("confirmatory") or {}
    ps = conf.get("path_summary") or {}
    n_seeds = ps.get("n_seeds")
    if int(n_seeds or 0) != int(require_n_seeds):
        errors.append(f"n_seeds_mismatch: got {n_seeds!r} want {require_n_seeds}")
    want_seeds = {str(i) for i in range(require_n_seeds)}
    per_seed = ps.get("per_seed")
    if isinstance(per_seed, dict):
        if set(map(str, per_seed.keys())) != want_seeds:
            errors.append(
                f"per_seed_keys: got {sorted(per_seed.keys())} want 0..{require_n_seeds - 1}"
            )
    elif isinstance(per_seed, list):
        if len(per_seed) != int(require_n_seeds):
            errors.append(
                f"per_seed_len: got {len(per_seed)} want {require_n_seeds}"
            )
        elif not all(_finite(x) for x in per_seed):
            errors.append("per_seed contains non-finite values")
    else:
        errors.append(f"per_seed_type: {type(per_seed).__name__}")

    if results.get("universe_arm") not in (None, "dyn_hrp"):
        errors.append(f"universe_arm: {results.get('universe_arm')!r} (expected dyn_hrp)")
    if results.get("capital_claim_allowed") is True:
        errors.append("legacy_capital_claim_allowed_true_refused")
    if results.get("tradable_claim_allowed") is True:
        errors.append("legacy_tradable_claim_allowed_true_refused")
    if results.get("capital_eligible") is True:
        errors.append("legacy_capital_eligible_true_refused")

    gates = conf.get("gates") or {}
    for g in ("gate1", "gate2", "gate3"):
        if g not in gates:
            errors.append(f"missing confirmatory.gates.{g}")
        elif "pass" not in (gates.get(g) or {}):
            errors.append(f"missing confirmatory.gates.{g}.pass")

    sharpe_mean = ps.get("sharpe_mean")
    if not _finite(sharpe_mean):
        errors.append("confirmatory.path_summary.sharpe_mean not finite")

    bench = conf.get("benchmark_sharpes") or results.get("benchmark_sharpes") or {}
    ew = bench.get("equal_weight")
    if ew is not None and not _finite(ew):
        errors.append("benchmark_sharpes.equal_weight not finite")

    stats = conf.get("stats_table") or results.get("stats_table") or {}
    if "deflated_sharpe" not in stats:
        errors.append("missing deflated_sharpe")
    if "hansen_spa_vs_ew" not in stats and "hansen_spa_pvalue" not in stats:
        errors.append("missing hansen_spa metric")

    neg = conf.get("negative_controls") or results.get("negative_controls") or {}
    if not neg:
        errors.append("missing negative_controls")
    if neg.get("pipeline_broken") is True:
        # Science finding: disclose in prose; do not block numbers rebuild.
        warnings.append("negative_controls.pipeline_broken is true (disclose)")
    for key in ("shuffled", "permuted", "date_shifted", "shuffled_labels", "permuted_signals", "date_shifted_signals"):
        sub = neg.get(key) or (neg.get("checks") or {}).get(key) or {}
        sh = sub.get("sharpe")
        if sh is not None and not _finite(sh):
            errors.append(f"negative_controls.{key}.sharpe not finite")

    pb = results.get("policy_behavior") or {}
    if not pb:
        warnings.append("policy_behavior missing (figures may skip)")
    elif pb.get("feeds_capital_gates") is True:
        errors.append("policy_behavior.feeds_capital_gates must be false")

    sharpe_std = ps.get("sharpe_std")
    if _finite(sharpe_std) and float(sharpe_std) < 0.01:
        warnings.append("near_zero_sharpe_std: disclose turnover binding in prose")

    book = results.get("book") or {}
    if results.get("book_error"):
        errors.append(f"book_error: {results.get('book_error')}")
    elif not book.get("out_dir"):
        warnings.append("book.out_dir missing")

    return {
        "ok": not errors,
        "label": label,
        "path": str(path),
        "k": k,
        "n_seeds": n_seeds,
        "gates": {g: (gates.get(g) or {}).get("pass") for g in ("gate1", "gate2", "gate3")},
        "errors": errors,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head", type=Path, default=DEFAULT_HEAD)
    p.add_argument("--surf-off", type=Path, default=DEFAULT_SURF_OFF)
    p.add_argument("--require-k", type=int, default=100)
    p.add_argument("--require-n-seeds", type=int, default=10)
    p.add_argument("--surf-off-optional", action="store_true")
    args = p.parse_args(argv)

    head = validate_summary(
        args.head,
        label="HEAD-EQ",
        require_k=args.require_k,
        require_n_seeds=args.require_n_seeds,
    )
    surf = validate_summary(
        args.surf_off,
        label="HEAD-SURF-OFF",
        require_k=args.require_k,
        require_n_seeds=args.require_n_seeds,
    )
    if args.surf_off_optional and not Path(args.surf_off).is_file():
        surf = {"ok": True, "label": "HEAD-SURF-OFF", "skipped": True, "path": str(args.surf_off)}

    report = {"head_eq": head, "head_surf_off": surf, "ok": head.get("ok") and surf.get("ok")}
    out = ROOT / "logs" / "artifacts" / "eq_alloc" / "headline_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
