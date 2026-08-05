#!/usr/bin/env python3
"""Cluster policy_behavior vectors; map cells to archetype animal mascots."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mascotrl.reporting.behavior_metrics import BEHAVIOUR_MEASURE_IDS
from mascotrl.reporting.policy_behavior import ARCHETYPE_SCORE_WEIGHTS

# Frozen archetype -> animal mascot (public codename layer).
ARCHETYPE_TO_ANIMAL: dict[str, str] = {
    "trend_follower": "Cheetah",
    "contrarian": "Fox",
    "risk_manager": "Tortoise",
    "speculator": "Magpie",
    "tactical_rotator": "Hummingbird",
    "mixed": "Owl",
}

CLUSTER_CODENAMES = (
    "Anchor",
    "Scout",
    "Bulwark",
    "Sprinter",
    "Weaver",
    "Hedgehog",
    "Compass",
    "Forge",
)

IMPUTATION_FAIL_FRACTION = 0.5

# Inflation mandate is identity without term_spread_z (cherrypick panel never
# attaches fioracle macro). Exclude from clustering so it does not dilute
# personality panel z-scores as a duplicate of the shared baseline.
ARCHETYPE_PANEL_EXCLUDE_SUBSTRINGS = ("pm-archetype_inflation",)


def _vector_from_behavior(beh: dict[str, Any]) -> tuple[np.ndarray, float]:
    """Return (vector, imputation_fraction) for missing/non-finite measures.

    Prefer ``archetype_composition`` rows when present so clustering follows
    the AA mixture rather than the raw 34-measure sleeve vector.
    """
    comp = beh.get("archetype_composition")
    if isinstance(comp, dict) and comp:
        # Stable order: frozen ARCHETYPE_SCORE_WEIGHTS keys, then extras.
        keys = list(ARCHETYPE_SCORE_WEIGHTS.keys())
        for k in sorted(comp.keys()):
            if k not in keys:
                keys.append(k)
        out: list[float] = []
        imputed = 0
        for key in keys:
            try:
                fv = float(comp.get(key, 0.0))
            except (TypeError, ValueError):
                fv = 0.0
                imputed += 1
            if not np.isfinite(fv):
                fv = 0.0
                imputed += 1
            out.append(fv)
        total = len(keys) or 1
        return np.asarray(out, dtype=np.float64), float(imputed) / float(total)

    b = beh.get("behaviour") or beh.get("behavior") or {}
    if not isinstance(b, dict):
        dim = len(BEHAVIOUR_MEASURE_IDS)
        return np.zeros(dim, dtype=np.float64), 1.0
    out = []
    imputed = 0
    total = len(BEHAVIOUR_MEASURE_IDS)
    for key in BEHAVIOUR_MEASURE_IDS:
        val = b.get(key)
        if val is None:
            imputed += 1
            fv = 0.0
        else:
            try:
                fv = float(val)
            except (TypeError, ValueError):
                imputed += 1
                fv = 0.0
            else:
                if not np.isfinite(fv):
                    imputed += 1
                    fv = 0.0
        out.append(fv)
    frac = float(imputed) / float(total) if total else 1.0
    return np.asarray(out, dtype=np.float64), frac


def _primary_from_beh(beh: dict[str, Any]) -> str:
    """Prefer composition argmax; fall back to legacy primary / name."""
    comp = beh.get("archetype_composition")
    if isinstance(comp, dict) and comp:
        try:
            return str(max(comp, key=lambda k: float(comp.get(k) or 0.0)))
        except (TypeError, ValueError):
            pass
    return str(
        beh.get("archetype_primary")
        or (beh.get("archetype") or {}).get("name")
        or "mixed"
    )

def _top_driving_measures(beh: dict[str, Any], arch: str, n: int = 2) -> list[str]:
    weights = ARCHETYPE_SCORE_WEIGHTS.get(arch) or {}
    if not weights:
        return []
    b = beh.get("behaviour") or {}
    scored = []
    for feat, w in weights.items():
        key = feat[len("neg_") :] if feat.startswith("neg_") else feat
        raw = b.get(key)
        try:
            fv = float(raw)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(fv):
            continue
        scored.append((abs(float(w) * fv), str(key)))
    scored.sort(reverse=True)
    return [name for _, name in scored[:n]]


def _why_string(beh: dict[str, Any]) -> str:
    arch = _primary_from_beh(beh)
    animal = ARCHETYPE_TO_ANIMAL.get(arch, "Owl")
    conf = beh.get("archetype_confidence")
    drivers = _top_driving_measures(beh, arch if arch != "mixed" else str(beh.get("archetype_top") or ""))
    if isinstance(beh.get("archetype_composition"), dict):
        top = sorted(
            beh["archetype_composition"].items(),
            key=lambda kv: float(kv[1]),
            reverse=True,
        )[:2]
        mix = ", ".join(f"{n.replace('_', ' ')} {float(v):.0%}" for n, v in top)
        return (
            f"Primary archetype {arch.replace('_', ' ')} ({animal}); "
            f"composition {mix}"
            + (f"; confidence {float(conf):.2f}" if conf is not None else "")
            + "."
        )
    if drivers:
        return (
            f"Primary archetype {arch.replace('_', ' ')} ({animal}); "
            f"top drivers: {', '.join(d.replace('_', ' ') for d in drivers)}."
        )
    margin = beh.get("archetype_margin")
    return (
        f"Primary archetype {arch.replace('_', ' ')} ({animal})"
        + (f"; margin to runner-up {float(margin):.2f}" if margin is not None else "")
        + "."
    )


def _infer_manifest_stems(manifest: dict[str, Any], target_dir: Path) -> list[str]:
    """Pick served stem list from manifest shape (cells / pick2 / k200)."""
    path_s = str(target_dir).lower()
    if "k200" in path_s:
        cells = (manifest.get("k200") or {}).get("cells") or []
    elif "narrative" in path_s or "pick2" in path_s:
        cells = (manifest.get("pick2") or {}).get("cells") or []
    else:
        cells = manifest.get("cells") or []
    if not isinstance(cells, list):
        return []
    dropped: set[str] = set()
    for item in manifest.get("dropped_cells") or []:
        if isinstance(item, dict):
            stem = str(item.get("stem") or "").strip()
        else:
            stem = str(item).strip()
        if stem:
            dropped.add(stem)
    return [str(c).strip() for c in cells if str(c).strip() and str(c).strip() not in dropped]


def _check_expect_stems(manifest_path: Path, target_dir: Path) -> tuple[bool, str]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"manifest_unreadable: {exc}"
    stems = _infer_manifest_stems(manifest, target_dir)
    if not stems:
        return False, "manifest_has_no_served_stems_for_target_dir"
    missing: list[str] = []
    for stem in stems:
        p = target_dir / f"{stem}_policy_behavior.json"
        if not p.is_file():
            missing.append(stem)
            continue
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            missing.append(stem)
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        return False, f"missing_or_unreadable_behavior_files: {preview}{suffix}"
    return True, ""


def cluster_behaviours(paths: list[Path], *, k: int = 4) -> dict[str, Any]:
    rows: list[tuple[str, dict[str, Any], np.ndarray, float, Path]] = []
    per_cell_imputation: dict[str, float] = {}
    for p in paths:
        try:
            beh = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        vec, impute_frac = _vector_from_behavior(beh)
        stem = p.stem.replace("_policy_behavior", "")
        per_cell_imputation[stem] = impute_frac
        rows.append((p.stem, beh, vec, impute_frac, p.parent))
    if not rows:
        return {"ok": False, "reason": "no_behavior_files", "clusters": []}

    max_impute = max(per_cell_imputation.values())
    imputation_meta = {
        "threshold": IMPUTATION_FAIL_FRACTION,
        "max_fraction": max_impute,
        "per_cell": per_cell_imputation,
    }
    if max_impute > IMPUTATION_FAIL_FRACTION:
        return {
            "ok": False,
            "reason": "imputation_exceeds_threshold",
            "clusters": [],
            "meta": {"imputation": imputation_meta},
        }

    # Pad short composition vectors to a common width.
    widths = [int(v.reshape(-1).shape[0]) for _, _, v, _, _ in rows]
    dim = max(widths) if widths else len(BEHAVIOUR_MEASURE_IDS)
    padded = []
    for _, _, v, _, _ in rows:
        vv = np.asarray(v, dtype=np.float64).reshape(-1)
        if vv.shape[0] < dim:
            vv = np.pad(vv, (0, dim - vv.shape[0]))
        padded.append(vv[:dim])
    X = np.stack(padded)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    X = (X - mu) / sd
    rng = np.random.default_rng(0)
    k = min(int(k), len(rows))
    centers = X[rng.choice(len(rows), size=k, replace=False)]
    labels = np.zeros(len(rows), dtype=int)
    for _ in range(20):
        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        labels = dists.argmin(axis=1)
        for i in range(k):
            mask = labels == i
            if mask.any():
                centers[i] = X[mask].mean(axis=0)

    cells_out: list[dict[str, Any]] = []
    for name, beh, _vec, impute_frac, parent in rows:
        arch = _primary_from_beh(beh)
        stem = name.replace("_policy_behavior", "")
        interp_path = parent / f"{stem}_interpretability.json"
        cell_row = {
            "cell_id": name,
            "archetype_primary": arch,
            "animal_mascot": ARCHETYPE_TO_ANIMAL.get(arch, "Owl"),
            "archetype_scores": beh.get("archetype_scores") or {},
            "imputation_fraction": impute_frac,
            "why": _why_string(beh),
            "interpretability_available": interp_path.is_file(),
        }
        if isinstance(beh.get("archetype_composition"), dict):
            cell_row["archetype_composition"] = dict(beh["archetype_composition"])
            cell_row["archetype_confidence"] = beh.get("archetype_confidence")
        cells_out.append(cell_row)

    clusters = []
    for i in range(k):
        members = [rows[j][0] for j in range(len(rows)) if labels[j] == i]
        archs = [
            _primary_from_beh(rows[j][1])
            for j in range(len(rows))
            if labels[j] == i
        ]
        interp_members = sum(
            1
            for j in range(len(rows))
            if labels[j] == i
            and (
                rows[j][4]
                / f"{rows[j][0].replace('_policy_behavior', '')}_interpretability.json"
            ).is_file()
        )
        dominant = max(set(archs), key=archs.count) if archs else "mixed"
        clusters.append(
            {
                "codename": CLUSTER_CODENAMES[i % len(CLUSTER_CODENAMES)],
                "cluster_id": i,
                "dominant_archetype": dominant,
                "animal_mascot": ARCHETYPE_TO_ANIMAL.get(dominant, "Owl"),
                "n_members": len(members),
                "members": members,
                "interpretability_n": interp_members,
            }
        )

    return {
        "ok": True,
        "clusters": clusters,
        "cells": cells_out,
        "meta": {"imputation": imputation_meta},
        "appendix": [
            {
                "codename": c["codename"],
                "animal_mascot": c["animal_mascot"],
                "dominant_archetype": c["dominant_archetype"],
                "why": (
                    f"Cluster {c['cluster_id']} groups {c['n_members']} cells; "
                    f"dominant archetype {c['dominant_archetype']} ({c['animal_mascot']})."
                ),
                "cells": c["members"],
            }
            for c in clusters
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dir",
        type=Path,
        default=ROOT / "logs/artifacts/spectrum/fullgrid",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "logs/artifacts/spectrum/fullgrid/behaviour_codenames.json",
    )
    p.add_argument("--k", type=int, default=4)
    p.add_argument(
        "--expect-stems",
        type=Path,
        default=None,
        metavar="MANIFEST.json",
        help="Refuse unless every served stem for this wave dir has a readable *_policy_behavior.json",
    )
    args = p.parse_args(argv)

    if args.expect_stems is not None:
        ok_stems, reason = _check_expect_stems(args.expect_stems, args.dir)
        if not ok_stems:
            print(f"assign_behavior_codenames: {reason}", file=sys.stderr)
            return 1

    paths = sorted(args.dir.glob("*_policy_behavior.json"))
    paths = [
        p
        for p in paths
        if not any(s in p.stem for s in ARCHETYPE_PANEL_EXCLUDE_SUBSTRINGS)
    ]
    payload = cluster_behaviours(paths, k=int(args.k))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ok={payload.get('ok')}")
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
