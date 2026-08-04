"""Load spectrum / arm artifacts from per-arm and flat artifact roots."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ARM_IDS = ("opt", "eq", "mix")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return blob if isinstance(blob, dict) else None


def resolve_artifact(
    name: str,
    *,
    arms_root: Path | None = None,
    arm: str | None = None,
    artifacts_flat: Path | None = None,
) -> Path | None:
    """Prefer ``arms/{arm}/{name}``, then flat ``artifacts/{name}``."""
    candidates: list[Path] = []
    if arms_root is not None and arm is not None:
        candidates.append(Path(arms_root) / arm / name)
    if artifacts_flat is not None:
        candidates.append(Path(artifacts_flat) / name)
    if arms_root is not None and arm is None:
        # Prefer opt as legacy stand-in when arm unspecified.
        candidates.append(Path(arms_root) / "opt" / name)
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_json_artifact(
    name: str,
    *,
    arms_root: Path | None = None,
    arm: str | None = None,
    artifacts_flat: Path | None = None,
) -> dict[str, Any] | None:
    path = resolve_artifact(
        name, arms_root=arms_root, arm=arm, artifacts_flat=artifacts_flat
    )
    return _read_json(path) if path is not None else None


def load_cpcv(
    arm: str,
    *,
    arms_root: Path,
    artifacts_flat: Path | None = None,
) -> dict[str, Any] | None:
    return load_json_artifact(
        "cpcv_path_summary.json",
        arms_root=arms_root,
        arm=arm,
        artifacts_flat=artifacts_flat if arm == "opt" else None,
    )


def load_gate3(
    arm: str,
    *,
    arms_root: Path,
    artifacts_flat: Path | None = None,
) -> dict[str, Any] | None:
    for name in ("gate3_same_fold.json", "gate3_baselines.json"):
        blob = load_json_artifact(
            name,
            arms_root=arms_root,
            arm=arm,
            artifacts_flat=artifacts_flat if arm == "opt" else None,
        )
        if blob is not None:
            return blob
    return None


def load_attrition(
    *,
    arms_root: Path | None = None,
    artifacts_flat: Path | None = None,
) -> dict[str, Any] | None:
    blob = load_json_artifact(
        "filter_attrition.json",
        arms_root=arms_root,
        arm="opt",
        artifacts_flat=artifacts_flat,
    )
    if blob is not None:
        return blob
    return load_json_artifact(
        "filter_attrition.json",
        arms_root=None,
        arm=None,
        artifacts_flat=artifacts_flat,
    )


def load_spectrum_summary(
    *,
    arms_root: Path | None = None,
    artifacts_flat: Path | None = None,
) -> dict[str, Any] | None:
    if artifacts_flat is not None:
        blob = _read_json(Path(artifacts_flat) / "spectrum_summary.json")
        if blob is not None:
            return blob
    if arms_root is not None:
        parent = Path(arms_root).parent
        blob = _read_json(parent / "spectrum_summary.json")
        if blob is not None:
            return blob
    return None


def path_summary(cpcv: dict[str, Any] | None) -> dict[str, Any]:
    if not cpcv:
        return {}
    ps = cpcv.get("path_summary")
    if isinstance(ps, dict) and ps:
        return ps
    nested = cpcv.get("cpcv") or {}
    if isinstance(nested, dict):
        ps = nested.get("path_summary")
        if isinstance(ps, dict):
            return ps
    return {}


def cpcv_paths(cpcv: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not cpcv:
        return []
    paths = cpcv.get("paths")
    if isinstance(paths, list) and paths:
        return [p for p in paths if isinstance(p, dict)]
    nested = cpcv.get("cpcv") or {}
    if isinstance(nested, dict):
        paths = nested.get("paths")
        if isinstance(paths, list):
            return [p for p in paths if isinstance(p, dict)]
    return []


def finetune_passes(cpcv: dict[str, Any] | None) -> int:
    if not cpcv:
        return 0
    cfg = cpcv.get("cpcv_config") or {}
    try:
        return int(cfg.get("finetune_passes") or 0)
    except (TypeError, ValueError):
        return 0


def break_even(cpcv: dict[str, Any] | None) -> float | None:
    """Return finite break-even multiplier, else None (undefined)."""
    if not cpcv:
        return None
    for src in (cpcv.get("gate1") or {}, cpcv.get("cost_ladder") or {}):
        if not isinstance(src, dict):
            continue
        be = src.get("break_even_spread_multiplier")
        if be is None:
            continue
        try:
            x = float(be)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            return x
    return None


def gate3_baselines(gate3: dict[str, Any] | None) -> dict[str, float]:
    """Map baseline name -> Sharpe from a gate3 artifact."""
    if not gate3:
        return {}
    g = gate3.get("gate3") if isinstance(gate3.get("gate3"), dict) else gate3
    bas = (g or {}).get("baselines") or {}
    out: dict[str, float] = {}
    if not isinstance(bas, dict):
        return out
    for name, payload in bas.items():
        if not isinstance(payload, dict):
            continue
        sh = payload.get("sharpe")
        if sh is None:
            continue
        try:
            x = float(sh)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            out[str(name)] = x
    return out


def discover_arms(arms_root: Path) -> list[str]:
    root = Path(arms_root)
    found = [a for a in ARM_IDS if (root / a).is_dir()]
    return found if found else list(ARM_IDS)
