#!/usr/bin/env python3
"""Econometric/behavioural sanity battery for the VAL validation wave."""
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

from scripts.validate_remote_cell import validate_remote_cell

HYBRID_STEM = "eq_K100_single_ppo_mlp_softmax_mean_std_cao_tw-hybrid_pretrain_finetune"
MAMBA_PROBE_PREFIXES = ("eq_K10_single_ppo_mamba_", "eq_K25_single_ppo_mamba_")


def _k_from_stem(stem: str) -> int:
    """Parse K from spectrum cell id (e.g. eq_K100_... -> 100)."""
    if "_K" not in stem:
        return 0
    try:
        return int(stem.split("_K", 1)[1].split("_", 1)[0])
    except (TypeError, ValueError, IndexError):
        return 0


def _load_manifest(root: Path) -> dict:
    path = root / "config" / "spectrum" / "cherrypick_val" / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing VAL manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _entry_for_stem(manifest: dict, stem: str) -> dict | None:
    for e in manifest.get("entries") or []:
        if str(e.get("stem")) == stem:
            return e
    return None


def _check_weights(artifact: dict, *, tol: float = 1e-4) -> list[str]:
    errors: list[str] = []
    head = str(artifact.get("weight_head") or artifact.get("head_axis_id") or "")
    weights = artifact.get("final_weights") or artifact.get("weights")
    if weights is None:
        return errors
    if not isinstance(weights, list):
        return errors
    flat = []
    for row in weights:
        if isinstance(row, list):
            flat.extend(float(x) for x in row)
        else:
            flat.append(float(row))
    if not flat:
        return errors
    if "softmax" in head or "dirichlet" in head:
        s = sum(flat)
        if abs(s - 1.0) > tol:
            errors.append(f"weights_sum={s:.6f} expected~1.0")
        if any(x < -tol for x in flat):
            errors.append("negative weight in long-only head")
    if "tanh_l1" in head:
        l1 = sum(abs(x) for x in flat)
        if l1 > 1.0 + tol:
            errors.append(f"tanh_l1 L1={l1:.6f} > 1")
    return errors


def validate_val_artifact(
    stem: str,
    artifact: dict,
    *,
    manifest: dict,
    expected_digest: str | None = None,
) -> dict[str, Any]:
    """Return {ok, errors, stem, role}."""
    errors: list[str] = []
    entry = _entry_for_stem(manifest, stem) or {}
    role = str(entry.get("role") or "")

    val = validate_remote_cell(artifact, expected_container_digest=expected_digest)
    if not val.get("ok"):
        errors.extend(str(e) for e in (val.get("errors") or []))

    # Legacy refuse: if an old artifact still carries claim keys as True, fail.
    for key in ("capital_claim_allowed", "tradable_claim_allowed", "capital_eligible"):
        if artifact.get(key) is True:
            errors.append(f"legacy_{key}_true_refused")
    if artifact.get("dry_run") or artifact.get("toy_panel"):
        errors.append("dry_run/toy_panel refused")

    panel_src = str(
        artifact.get("panel_source")
        or (artifact.get("runner_artifact") or {}).get("panel_source")
        or ""
    )
    # lake_sp500_sec = lake-backed equity spine; optionmetrics = CPCV default
    # stamp used by physics train_world transfer paths (same eq OOS substrate).
    if panel_src and panel_src not in {"lake_sp500_sec", "optionmetrics"}:
        errors.append(f"unexpected panel_source={panel_src!r}")

    cpcv = artifact.get("cpcv") or {}
    if not cpcv:
        # Spectrum cells often nest CPCV knobs under spectrum_budget.
        budget = artifact.get("spectrum_budget") or {}
        if budget.get("cpcv_n_splits") is not None:
            cpcv = {
                "n_splits": budget.get("cpcv_n_splits"),
                "n_test_groups": budget.get("cpcv_n_test_groups"),
                "purge_days": artifact.get("cpcv_purge_days")
                or (artifact.get("resolved") or {}).get("cpcv_purge_days")
                or 21,
                "embargo_days": artifact.get("cpcv_embargo_days")
                or (artifact.get("resolved") or {}).get("cpcv_embargo_days")
                or 21,
            }
    if cpcv:
        if int(cpcv.get("n_splits") or 0) not in {6, 8}:
            errors.append(f"unexpected cpcv n_splits={cpcv.get('n_splits')}")
        if int(cpcv.get("purge_days") or 0) != 21:
            errors.append("purge_days != 21")
        if int(cpcv.get("embargo_days") or 0) != 21:
            errors.append("embargo_days != 21")

    resolved = artifact.get("resolved") or {}
    algo = str(artifact.get("algo") or resolved.get("algo") or "")
    claim_tier = str(
        artifact.get("claim_tier")
        or (artifact.get("spectrum_budget") or {}).get("claim_tier")
        or ""
    )
    if algo == "happo":
        if claim_tier not in {"dispatch_only", "research"}:
            errors.append(f"unexpected happo claim_tier={claim_tier!r}")

    if stem.startswith(MAMBA_PROBE_PREFIXES):
        k = int(
            artifact.get("k")
            or artifact.get("n_assets")
            or resolved.get("k")
            or resolved.get("n_assets")
            or 0
        )
        if k not in {10, 25}:
            k = _k_from_stem(stem)
        if k not in {10, 25}:
            errors.append(f"mamba probe k={k} not in {{10,25}}")

    errors.extend(_check_weights(artifact))

    metrics = artifact.get("metrics") or {}
    for key, val_m in metrics.items():
        if isinstance(val_m, (int, float)) and not math.isfinite(float(val_m)):
            errors.append(f"non-finite metric {key}={val_m}")

    return {"ok": not errors, "errors": errors, "stem": stem, "role": role}


def validate_hybrid_error(error_doc: dict) -> dict[str, Any]:
    errors: list[str] = []
    reason = str(error_doc.get("reason") or error_doc.get("error") or "")
    n_att = int(
        error_doc.get("n_attempts")
        or error_doc.get("n_logical_attempts")
        or error_doc.get("attempts")
        or 0
    )
    if n_att > 1:
        errors.append(f"hybrid error n_attempts={n_att} > 1")
    blob = json.dumps(error_doc).lower()
    if "non-finite" not in blob and "nan" not in blob and "policy_obs" not in blob:
        errors.append("hybrid error missing non-finite policy_obs signature")
    return {"ok": not errors, "errors": errors, "stem": HYBRID_STEM, "role": "hybrid_negative_control"}


def validate_val_dir(
    artifact_dir: Path,
    *,
    root: Path | None = None,
    expected_digest: str | None = None,
) -> dict[str, Any]:
    base = root or ROOT
    manifest = _load_manifest(base)
    want = set(manifest.get("cells") or [])
    anomalies: list[dict] = []
    passed: list[str] = []

    for stem in sorted(want):
        if stem == HYBRID_STEM:
            final = artifact_dir / f"{stem}.json"
            err = artifact_dir / f"{stem}.error.json"
            if final.is_file():
                art = json.loads(final.read_text(encoding="utf-8"))
                res = validate_val_artifact(
                    stem, art, manifest=manifest, expected_digest=expected_digest
                )
            elif err.is_file():
                res = validate_hybrid_error(json.loads(err.read_text(encoding="utf-8")))
            else:
                res = {"ok": False, "errors": ["missing final and error"], "stem": stem}
        else:
            final = artifact_dir / f"{stem}.json"
            if not final.is_file():
                res = {"ok": False, "errors": ["missing final"], "stem": stem}
            else:
                art = json.loads(final.read_text(encoding="utf-8"))
                res = validate_val_artifact(
                    stem, art, manifest=manifest, expected_digest=expected_digest
                )
        if res.get("ok"):
            passed.append(stem)
        else:
            anomalies.append(res)

    ok = len(anomalies) == 0 and len(passed) == len(want)
    return {
        "ok": ok,
        "n_expected": len(want),
        "n_passed": len(passed),
        "n_anomalies": len(anomalies),
        "anomalies": anomalies,
        "passed": passed,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dir",
        type=Path,
        default=ROOT / "logs/artifacts/spectrum/cherrypick_val",
    )
    p.add_argument("--expected-digest", default="")
    args = p.parse_args(argv)
    result = validate_val_dir(
        args.dir,
        expected_digest=args.expected_digest or None,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
