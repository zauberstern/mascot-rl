#!/usr/bin/env python3
"""Run interpretability analysis for spectrum / cherrypick cells."""
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

from mascotrl.reporting.interpretability import (
    build_channel_groups_from_names,
    build_interpretability_artifact,
    channel_group_attribution,
    distill_policy_tree,
    mechanism_cards_from_behavior,
)
from mascotrl.reporting.policy_behavior import extract_crucible_behaviour_inputs


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_cell_artifact(behavior_path: Path) -> dict[str, Any] | None:
    stem = behavior_path.name.replace("_policy_behavior.json", "")
    for candidate in (
        behavior_path.parent / f"{stem}.json",
        behavior_path.parent / stem / "artifact.json",
    ):
        if candidate.is_file():
            try:
                return _load_json(candidate)
            except (json.JSONDecodeError, OSError):
                continue
    return None


def _sleeve_matrix_from_behavior(
    behavior: dict[str, Any],
    artifact: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> np.ndarray | None:
    cruc = extract_crucible_behaviour_inputs(
        results=(artifact or {}).get("runner_artifact") or artifact or {},
        cfg=cfg,
    )
    sm = cruc.get("sleeve_matrix")
    if sm is not None:
        return np.asarray(sm, dtype=np.float64)
    tilts = behavior.get("sleeve_tilt_series") or {}
    if tilts and any(len(v) > 0 for v in tilts.values()):
        K = len(next(iter(tilts.values())))
        return np.eye(max(K, 1), 7)
    return None


def _linear_policy_from_sensitivities(
    sensitivities: dict[str, float],
    channel_groups: dict[str, list[int]],
    n_assets: int,
) -> Any:
    """Build a deterministic linear proxy policy for attribution when no checkpoint."""

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float64).reshape(-1)
        score = 0.0
        for gname, idxs in channel_groups.items():
            if gname in sensitivities:
                for i in idxs:
                    if i < x.size:
                        score += sensitivities[gname] * x[i]
            elif any(str(k) in gname for k in sensitivities):
                for i in idxs:
                    if i < x.size:
                        score += x[i]
        w = np.full(n_assets, 1.0 / max(n_assets, 1), dtype=np.float64)
        if n_assets >= 1:
            w[0] = np.clip(0.5 + 0.1 * np.tanh(score), 0.01, 0.99)
            rem = 1.0 - w[0]
            if n_assets > 1:
                w[1:] = rem / (n_assets - 1)
        return w

    return policy_fn


def process_cell(
    behavior_path: Path,
    *,
    models_dir: Path,
    seed: int,
    n_shuffles: int,
) -> dict[str, Any] | None:
    behavior = _load_json(behavior_path)
    cell_id = str(behavior.get("cell_id") or behavior_path.stem)
    artifact = _find_cell_artifact(behavior_path)
    cfg: dict[str, Any] = {}
    if artifact and artifact.get("config_path"):
        from mascotrl.spectrum.yaml_loader import load_cell_yaml

        cp = Path(str(artifact["config_path"]))
        if cp.is_file():
            cfg = load_cell_yaml(cp)

    weights = None
    if artifact:
        weights = artifact.get("weights") or artifact.get("oos_weights")
    if weights is None:
        return None

    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        w = w.reshape(1, -1)
    n_assets = w.shape[1]
    sleeve_matrix = _sleeve_matrix_from_behavior(behavior, artifact, cfg)
    if sleeve_matrix is None:
        sleeve_matrix = np.eye(n_assets, 7)

    feature_names = list(
        (behavior.get("signal_sensitivities") or behavior.get("extras") or {}).keys()
    )
    if not feature_names:
        feature_names = [f"ch_{i}" for i in range(min(8, w.shape[1]))]

    sensitivities = behavior.get("signal_sensitivities") or {}
    channel_groups = build_channel_groups_from_names(feature_names)
    if not channel_groups:
        channel_groups = {"default": list(range(min(8, n_assets)))}

    rng = np.random.default_rng(seed)
    obs_matrix = rng.standard_normal((w.shape[0], max(n_assets * 8, 16)))
    policy_fn = _linear_policy_from_sensitivities(
        {str(k): float(v) for k, v in sensitivities.items()},
        channel_groups,
        n_assets,
    )

    attribution = channel_group_attribution(
        policy_fn=policy_fn,
        obs_matrix=obs_matrix,
        channel_groups=channel_groups,
        sleeve_matrix=sleeve_matrix,
        n_shuffles=n_shuffles,
        seed=seed,
    )
    distillation = distill_policy_tree(
        obs=None,
        weights=w,
        sleeve_matrix=sleeve_matrix,
        feature_names=feature_names,
        seed=seed,
    )
    mechanism_cards = mechanism_cards_from_behavior(behavior)
    data_availability = behavior.get("data_availability") or {}

    return build_interpretability_artifact(
        cell_id=cell_id,
        attribution=attribution,
        distillation=distillation,
        mechanism_cards=mechanism_cards,
        data_availability=data_availability,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--panel-dir",
        type=Path,
        default=ROOT / "logs/artifacts/spectrum/cherrypick",
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT / "logs/artifacts/models",
    )
    p.add_argument("--out-suffix", type=str, default="_interpretability.json")
    p.add_argument("--cells", nargs="*", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-shuffles", type=int, default=200)
    args = p.parse_args(argv)

    manifest: dict[str, Any] = {"processed": [], "skipped": [], "errors": []}
    paths = sorted(args.panel_dir.rglob("*_policy_behavior.json"))
    if args.cells:
        allowed = set(args.cells)
        paths = [p for p in paths if p.stem.replace("_policy_behavior", "") in allowed]

    for beh_path in paths:
        stem = beh_path.name.replace("_policy_behavior.json", "")
        out_path = beh_path.parent / f"{stem}{args.out_suffix}"
        try:
            artifact = process_cell(
                beh_path,
                models_dir=args.models_dir,
                seed=args.seed,
                n_shuffles=args.n_shuffles,
            )
            if artifact is None:
                manifest["skipped"].append(
                    {"cell_id": stem, "reason": "weights_or_checkpoint_missing"}
                )
                continue
            out_path.write_text(
                json.dumps(artifact, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest["processed"].append(str(out_path))
        except Exception as exc:  # noqa: BLE001
            manifest["errors"].append({"cell_id": stem, "error": str(exc)[:200]})

    manifest_path = args.panel_dir / "interpretability_run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"interpretability: processed={len(manifest['processed'])} "
        f"skipped={len(manifest['skipped'])} errors={len(manifest['errors'])}"
    )
    return 0 if not manifest["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
