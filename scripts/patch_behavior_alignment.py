#!/usr/bin/env python3
"""Patch alignment / softmax-exception fields onto existing behaviour exports.

Faster than full ``--refresh-behavior`` (which re-scores holdings/RBSA/turbulence).
Uses designed_personality + compute_personality_alignment against already-written
archetype scores. Safe to re-run; full refresh remains the complete path.
"""
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

from src.reporting.policy_behavior import (
    compute_personality_alignment,
    designed_personality,
    normalize_weight_head,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        from scripts.run_spectrum_campaign import load_cell_yaml

        return load_cell_yaml(path)
    except Exception:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve_cfg(stem: str, config_dir: Path, beh: dict[str, Any]) -> dict[str, Any]:
    matches = list(config_dir.rglob(f"{stem}.yaml"))
    if matches:
        return _load_yaml(matches[0])
    # Minimal cfg from behaviour + stem tokens.
    return {
        "algo": beh.get("algo"),
        "objective": beh.get("objective"),
        "weight_head": beh.get("weight_head"),
        "policy_mode": beh.get("policy_mode"),
    }


def _head_from(stem: str, cfg: dict[str, Any], beh: dict[str, Any]) -> str:
    raw = cfg.get("weight_head") or beh.get("weight_head") or ""
    head = normalize_weight_head(str(raw)) if raw else ""
    if head in ("", "single", "multi", "balanced", "shared", "long_only"):
        s = stem.lower()
        if "sparse_tilt" in s:
            return "sparse_tilt"
        if "tanh_l1" in s:
            return "tanh_l1"
        if "dirichlet" in s:
            return "dirichlet"
        if "softmax" in s:
            return "softmax"
        pm = str(cfg.get("policy_mode") or beh.get("policy_mode") or "")
        head = normalize_weight_head(pm)
    if head in ("", "single", "multi", "balanced", "shared", "long_only"):
        return "softmax"
    return head


def _mandate(cfg: dict[str, Any], stem: str) -> str:
    m = str(cfg.get("mandate_preset") or cfg.get("policy_mandate") or "").lower()
    if m:
        return m
    pm = str(cfg.get("policy_mode") or "").lower()
    if pm.startswith("archetype_"):
        return pm
    for tok in ("archetype_carry", "archetype_crisis", "archetype_inflation"):
        if tok in stem.lower():
            return tok
    return ""


def patch_one(path: Path, config_dir: Path) -> dict[str, Any]:
    beh = json.loads(path.read_text(encoding="utf-8"))
    stem = path.name.replace("_policy_behavior.json", "")
    cfg = _resolve_cfg(stem, config_dir, beh)
    head = _head_from(stem, cfg, beh)
    mandate = _mandate(cfg, stem)
    algo = str(cfg.get("algo") or beh.get("algo") or "").lower()
    objective = str(cfg.get("objective") or beh.get("objective") or "").lower()
    designed = designed_personality(
        objective=objective,
        algo=algo,
        weight_head=head,
        mandate_preset=mandate,
    )
    observed = {
        "archetype_primary": beh.get("archetype_primary"),
        "archetype_scores": beh.get("archetype_scores") or {},
        "behaviour": beh.get("behaviour") or {},
    }
    alignment = compute_personality_alignment(designed, observed)

    behaviour = dict(beh.get("behaviour") or {})
    try:
        l1 = float(behaviour.get("l1_vs_ew_mean", float("nan")))
    except (TypeError, ValueError):
        l1 = float("nan")
    if head == "softmax" and math.isfinite(l1) and l1 > 0.25:
        behaviour["softmax_collapse_exception"] = True
        behaviour["softmax_escape_note"] = (
            f"algo={algo} produces larger logit scale under softmax "
            f"(l1_vs_ew={l1:.4f}>0.25)"
        )
    else:
        behaviour.pop("softmax_collapse_exception", None)
        behaviour.pop("softmax_escape_note", None)

    beh["behaviour"] = behaviour
    beh["alignment_pass"] = bool(alignment["alignment_pass"])
    beh["alignment_score"] = float(alignment["alignment_score"])
    beh["designed_personality"] = str(alignment["designed_personality"])
    beh["observed_personality"] = str(alignment["observed_personality"])
    beh["alignment_divergence"] = str(alignment.get("divergence_explanation") or "")
    beh["weight_head"] = head
    if mandate:
        beh["mandate_preset"] = mandate
    path.write_text(json.dumps(beh, indent=2), encoding="utf-8")
    return {
        "stem": stem,
        "designed": beh["designed_personality"],
        "observed": beh["observed_personality"],
        "alignment_pass": beh["alignment_pass"],
        "head": head,
        "mandate": mandate or None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--config-dir", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for path in sorted(args.dir.glob("*_policy_behavior.json")):
        try:
            rows.append(patch_one(path, args.config_dir))
        except Exception as exc:  # noqa: BLE001
            rows.append({"stem": path.name, "error": str(exc)[:200]})
    n_pass = sum(1 for r in rows if r.get("alignment_pass") is True)
    n_fail = sum(1 for r in rows if r.get("alignment_pass") is False)
    print(
        json.dumps(
            {
                "n": len(rows),
                "alignment_pass": n_pass,
                "alignment_fail": n_fail,
                "errors": sum(1 for r in rows if "error" in r),
                "sample": rows[:5],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
