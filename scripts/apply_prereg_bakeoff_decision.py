#!/usr/bin/env python3
"""Apply PREREG_20260811 section 4 arm-selection rule to bakeoff summaries.

Reads each arm's ``cpcv_path_summary.json`` under the bakeoff root, evaluates
``confirmatory.decision_fields`` plus after-cost CPCV Sharpe, writes the winner
(or null → ``dyn_hrp`` reference) to the trial ledger and a decision JSON.
Does not mutate the spine YAML unless ``--promote-yaml`` is passed.

Null reference is ``dyn_hrp`` (best raw bakeoff Sharpe among current arms) while
still declaring ``status=null_reference`` when no arm clears §4 predicates.
``dyn_crucible`` remains evaluable but is stamped deferred for a later experiment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_ARMS = (
    "dyn_hrp",
    "dyn_liquidity",
    "dyn_crucible",
)
NULL_REFERENCE_ARM = "dyn_hrp"
DEFERRED_ARMS = ("dyn_crucible",)


def _finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return v


def normalize_decision_fields(
    raw: dict[str, Any],
    *,
    path_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map campaign decision_fields aliases onto the PREREG §4 predicate names."""
    path_summary = path_summary or {}
    fields = dict(raw)

    if "equal_weight_collapse_detected" not in fields:
        if "equal_weight_collapse_detected_any" in fields:
            fields["equal_weight_collapse_detected"] = bool(
                fields["equal_weight_collapse_detected_any"]
            )

    if fields.get("sharpe_std") is None:
        for key in ("sharpe_std_across_seeds", "cross_seed_sharpe_std"):
            if fields.get(key) is not None:
                fields["sharpe_std"] = fields[key]
                break
        if fields.get("sharpe_std") is None and path_summary.get("sharpe_std") is not None:
            fields["sharpe_std"] = path_summary.get("sharpe_std")

    if fields.get("turnover_cap_binding_fraction") is None:
        if fields.get("turnover_cap_binding_fraction_mean") is not None:
            fields["turnover_cap_binding_fraction"] = fields[
                "turnover_cap_binding_fraction_mean"
            ]

    return fields


def qualifies(fields: dict[str, Any], *, projection_mode: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    collapse = fields.get("equal_weight_collapse_detected")
    if collapse is not False:
        reasons.append(f"equal_weight_collapse_detected={collapse!r}")
    sharpe_std = _finite(fields.get("sharpe_std") or fields.get("cross_seed_sharpe_std"))
    if sharpe_std is None or sharpe_std <= 0.01:
        reasons.append(f"sharpe_std={sharpe_std!r} (need > 0.01)")
    if str(projection_mode).lower() == "hard":
        bind = _finite(fields.get("turnover_cap_binding_fraction"))
        if bind is None or not (0.05 < bind < 0.95):
            reasons.append(f"turnover_cap_binding_fraction={bind!r} (need (0.05, 0.95))")
    return (not reasons), reasons


def _projection_mode(blob: dict[str, Any]) -> str:
    cfg = blob.get("cfg")
    if isinstance(cfg, dict) and cfg.get("projection_mode") is not None:
        return str(cfg.get("projection_mode"))
    if blob.get("projection_mode") is not None:
        return str(blob.get("projection_mode"))
    # Campaign often stores cfg as a workflow path string.
    if isinstance(cfg, str) and cfg.strip():
        path = Path(cfg)
        if not path.is_file():
            path = ROOT / cfg
        if path.is_file():
            try:
                import yaml

                y = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                if y.get("projection_mode") is not None:
                    return str(y.get("projection_mode"))
            except Exception:
                pass
    return "hard"


def load_arm(summary_path: Path) -> dict[str, Any]:
    blob = json.loads(summary_path.read_text(encoding="utf-8"))
    conf = blob.get("confirmatory") or {}
    path_sum = conf.get("path_summary") or blob.get("path_summary") or {}
    fields = normalize_decision_fields(
        dict(conf.get("decision_fields") or {}),
        path_summary=dict(path_sum),
    )
    sharpe = _finite(path_sum.get("sharpe_mean"))
    projection = _projection_mode(blob)
    ok, reasons = qualifies(fields, projection_mode=projection)
    return {
        "arm": summary_path.parent.name,
        "path": str(summary_path),
        "sharpe_mean": sharpe,
        "decision_fields": fields,
        "projection_mode": projection,
        "qualifies": ok,
        "reject_reasons": reasons,
    }


def choose_winner(
    rows: list[dict[str, Any]],
    *,
    null_reference: str = NULL_REFERENCE_ARM,
    deferred_arms: tuple[str, ...] = DEFERRED_ARMS,
) -> dict[str, Any]:
    eligible = [r for r in rows if r.get("qualifies") and r.get("sharpe_mean") is not None]
    if not eligible:
        return {
            "winner": str(null_reference),
            "status": "null_reference",
            "promotable": False,
            "rule": (
                "PREREG_20260811 §4: no arm qualifies → freeze "
                f"{null_reference} reference and declare null"
            ),
            "eligible": [],
            "deferred_arms": [a for a in deferred_arms if a != null_reference],
            "note": (
                "Null reference is the preferred freeze among evaluated arms, "
                "not a §4-qualifying winner. Deferred arms may be re-run later "
                "as separate experiments without changing this seal path."
            ),
        }
    eligible.sort(key=lambda r: float(r["sharpe_mean"]), reverse=True)
    winner = eligible[0]
    return {
        "winner": winner["arm"],
        "status": "promotable",
        "promotable": True,
        "rule": "PREREG_20260811 §4: max after-cost CPCV Sharpe among qualifying arms",
        "eligible": [r["arm"] for r in eligible],
        "winner_sharpe_mean": winner["sharpe_mean"],
        "winner_decision_fields": winner["decision_fields"],
        "deferred_arms": [a for a in deferred_arms if a != winner["arm"]],
    }


def promote_yaml(yaml_path: Path, arm: str) -> None:
    import yaml

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    cfg["universe_arm"] = arm
    yaml_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bakeoff-root",
        type=Path,
        default=ROOT / "logs/artifacts/eq_alloc/bakeoff",
    )
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument(
        "--null-reference",
        default=NULL_REFERENCE_ARM,
        help="Arm to freeze when no §4-qualifying winner exists (default: dyn_hrp).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "logs/artifacts/eq_alloc/bakeoff/prereg_decision.json",
    )
    ap.add_argument(
        "--ledger",
        type=Path,
        default=ROOT / "logs/trial_ledger.json",
    )
    ap.add_argument("--promote-yaml", action="store_true")
    ap.add_argument(
        "--yaml",
        type=Path,
        default=ROOT / "config/workflows/arm_equity.yaml",
    )
    args = ap.parse_args(argv)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for arm in [a.strip() for a in str(args.arms).split(",") if a.strip()]:
        path = args.bakeoff_root / arm / "cpcv_path_summary.json"
        if not path.is_file():
            missing.append(arm)
            continue
        rows.append(load_arm(path))
    decision = choose_winner(
        rows,
        null_reference=str(args.null_reference),
        deferred_arms=DEFERRED_ARMS,
    )
    payload = {
        "schema": "mascotrl.prereg_bakeoff_decision.v1",
        "prereg": "PREREG_20260811",
        "arms_evaluated": rows,
        "missing_arms": missing,
        "decision": decision,
        "null_reference_arm": str(args.null_reference),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    from src.eval.pbo_appendix import append_trial_ledger_entry

    append_trial_ledger_entry(
        args.ledger,
        source="apply_prereg_bakeoff_decision",
        trial_id="bakeoff_prereg_decision_20260811",
        sharpe=decision.get("winner_sharpe_mean"),
        status=str(decision.get("status")),
        extra={
            "winner": decision.get("winner"),
            "eligible": decision.get("eligible"),
            "missing_arms": missing,
            "promotable": decision.get("promotable"),
            "deferred_arms": decision.get("deferred_arms"),
            "null_reference_arm": str(args.null_reference),
        },
    )
    if args.promote_yaml:
        promote_yaml(args.yaml, str(decision["winner"]))
        print(f"promoted universe_arm={decision['winner']!r} into {args.yaml}")
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
