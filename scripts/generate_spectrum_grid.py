#!/usr/bin/env python3
"""Generate protocol-parity spectrum fullgrid cells (WP-S4).

Legal count per (arm, K) = 754:
  legal (algo, body, objective) triples from ``validate_cfg``, then
  4 weight heads for continuous algos (incl. sparse_tilt) and 1 collapsed
  head for dqn/happo.

Writes ``config/spectrum/fullgrid/<cell_id>.yaml`` plus a refusal ledger.
Does not touch the legacy OFAT smoke cells under ``config/spectrum/*.yaml``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mascotrl.policy.rasp_locks import apply_rasp_defaults, assert_rasp_locks
from mascotrl.spectrum.capacity_probe import probe_universe_capacity
from mascotrl.spectrum.cell_schema import validate_cell_cfg
from mascotrl.spectrum.protocol_tiers import apply_protocol_tier
from mascotrl.spectrum.registry import (
    ALGO_HEADS,
    PORTFOLIO_ARM_IDS,
    _axis_id_to_weight_head,
    allowed_ids,
    validate_cfg,
)

OUT_DIR = ROOT / "config" / "spectrum" / "fullgrid"

# Protocol-parity base (replaces smoke OFAT _BASE). Soft OFAT stays legacy.
_BASE: dict[str, Any] = {
    "claim_tier": "research",
    "cost_in_decision": True,
    "reward_shaping_ablation": True,
    "primary_train": "historical_arm_env",
    "projection_mode": "hard",
    "turnover_limit": 0.15,
    "execution_impact_coef": 0.5,
    "rebalance_cadence": "monthly",
    "equity_bps": 5.0,
    "headline_fill": "pct75",
    "om_touch_enabled": True,
    "hedge_leg_spread_bps": 5.0,
    "execution_spread_bps": 5.0,
    "lr": 0.0003,
    "use_surface_signals": True,
    "selection_start": "2003-01-01",
    "selection_end": "2012-12-31",
    "oos_start": "2014-01-01",
    "oos_end": "2024-12-31",
    "scr_mix": "off",
    "weight_head_tilt_gain": 1.0,
    "grid_kind": "fullgrid",
}

# Imported SoT from registry (avoid duplicating head tables).
_ALGO_HEADS = ALGO_HEADS

EXPECTED_CELLS_PER_ARM_K = 754


def _weight_head_cfg(head_id: str) -> str:
    return _axis_id_to_weight_head(head_id)


def _cell_cfg(
    *,
    arm: str,
    k: int,
    algo: str,
    body: str,
    head: str,
    objective: str,
    tier: str,
    train_world: str = "historical",
    policy_mode: str = "shared",
) -> dict[str, Any]:
    agent = "multi" if algo == "happo" else "single"
    # Degenerate-head algos omit _{head} from the id (plan WP-S4).
    if algo in ("dqn", "happo"):
        cell_id = f"{arm}_K{k}_{agent}_{algo}_{body}_{objective}"
    else:
        cell_id = f"{arm}_K{k}_{agent}_{algo}_{body}_{head}_{objective}"
    cfg: dict[str, Any] = dict(_BASE)
    cfg.update(
        {
            "spectrum_cell_id": cell_id,
            "portfolio_arm": arm,
            "n_assets": int(k),
            "algo": algo,
            "policy_algo": algo,
            "architecture": body,
            "temporal_backend": body,
            "weight_head": _weight_head_cfg(head),
            "head_axis_id": head,
            "objective": objective,
            "train_world": train_world,
            "train_distribution": train_world,
            "policy_mode": policy_mode,
            "agent": agent,
            "policy": "happo" if algo == "happo" else "single_agent",
            "action_law": (
                "dirichlet_tilt"
                if head == "dirichlet_tilt"
                else (
                    "dirichlet_mean"
                    if head == "dirichlet_mean"
                    else (
                        "dirichlet_entropy"
                        if head == "dirichlet_entropy"
                        else head
                    )
                )
            ),
        }
    )
    apply_protocol_tier(cfg, tier)
    # Screening HAPPO is smoke-capped at runtime; stamp honesty into YAML.
    if algo == "happo" and str(tier).lower().strip() == "screening":
        cfg["happo_dispatch_only"] = True
    apply_rasp_defaults(cfg)
    return cfg


def enumerate_legal_cells(
    *,
    arms: tuple[str, ...] = PORTFOLIO_ARM_IDS,
    k_axis: tuple[int, ...] = (100,),
    tier: str = "screening",
    train_world: str = "historical",
    policy_mode: str = "shared",
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Emit legal fullgrid cells; refuse illegal triples into the ledger."""
    cells: list[dict[str, Any]] = []
    refused: list[dict[str, str]] = []
    algos = list(allowed_ids("algo"))
    bodies = list(allowed_ids("architecture"))
    objectives = list(allowed_ids("objective"))

    for arm in arms:
        for k in k_axis:
            for algo in algos:
                for body in bodies:
                    for objective in objectives:
                        probe = {
                            "algo": algo,
                            "architecture": body,
                            "objective": objective,
                            "train_world": train_world,
                            "policy_mode": policy_mode,
                            "projection_mode": "hard",
                            "turnover_limit": 0.15,
                        }
                        if body != "mlp":
                            probe["use_equity_feature_cube"] = True
                        try:
                            validate_cfg(probe)
                        except ValueError as exc:
                            reason = str(exc)
                            if "requires_episode_returns" in reason:
                                code = "objective_requires_episode_algo"
                            elif "requires_discrete" in reason:
                                code = "dqn_requires_mlp"
                            else:
                                code = "validate_cfg_refused"
                            refused.append(
                                {
                                    "arm": arm,
                                    "k": str(k),
                                    "algo": algo,
                                    "architecture": body,
                                    "objective": objective,
                                    "status": "refused",
                                    "skip_reason": code,
                                    "detail": reason.split(":", 1)[0][:120],
                                }
                            )
                            continue

                        for head in _ALGO_HEADS[algo]:
                            cfg = _cell_cfg(
                                arm=arm,
                                k=int(k),
                                algo=algo,
                                body=body,
                                head=head,
                                objective=objective,
                                tier=tier,
                                train_world=train_world,
                                policy_mode=policy_mode,
                            )
                            try:
                                # RASP locks on the concrete cell (dirichlet+dqn etc.).
                                assert_rasp_locks(cfg)
                                validate_cfg(cfg)
                            except ValueError as exc:
                                refused.append(
                                    {
                                        "cell_id": cfg["spectrum_cell_id"],
                                        "status": "refused",
                                        "skip_reason": str(exc).split(":", 1)[0],
                                    }
                                )
                                continue
                            # Schema must pass; generation fails closed on error.
                            validate_cell_cfg(cfg)
                            cells.append(cfg)
    return cells, refused


def count_legal_cells_per_arm_k() -> int:
    cells, _ = enumerate_legal_cells(arms=("eq",), k_axis=(100,), tier="screening")
    return len(cells)


def _yaml_dump(cfg: dict[str, Any]) -> str:
    lines = [
        f"# Protocol fullgrid cell: {cfg.get('spectrum_cell_id')}",
        "# Generated by scripts/generate_spectrum_grid.py; do not hand-edit.",
    ]
    for key, val in cfg.items():
        if key.startswith("_"):
            continue
        if isinstance(val, bool):
            lines.append(f"{key}: {'true' if val else 'false'}")
        elif isinstance(val, (list, tuple)):
            lines.append(f"{key}: [{', '.join(str(v) for v in val)}]")
        elif isinstance(val, float):
            lines.append(f"{key}: {val}")
        elif isinstance(val, int):
            lines.append(f"{key}: {val}")
        else:
            lines.append(f"{key}: {val}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--pool-size", type=int, default=480)
    p.add_argument("--tier", default="screening")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--arms", default="eq,opt,mix")
    p.add_argument("--k-axis", default="100,200", help="Comma K list; default 100,200")
    args = p.parse_args(argv)

    if str(args.k_axis).strip():
        k_axis = tuple(int(x) for x in str(args.k_axis).split(",") if x.strip())
        cap = probe_universe_capacity(int(args.pool_size), requested=k_axis)
        k_axis = cap.k_axis
    else:
        cap = probe_universe_capacity(int(args.pool_size), requested=(100, 200))
        k_axis = cap.k_axis

    arms = tuple(a.strip() for a in str(args.arms).split(",") if a.strip())
    cells, refused = enumerate_legal_cells(
        arms=arms, k_axis=k_axis, tier=str(args.tier)
    )
    per = count_legal_cells_per_arm_k()
    if per != EXPECTED_CELLS_PER_ARM_K:
        raise SystemExit(
            f"legal count drift: per_arm_k={per} expected={EXPECTED_CELLS_PER_ARM_K}"
        )

    out_dir: Path = args.out_dir
    if not args.dry_run:
        # Clear previous fullgrid yaml so stale ids do not linger.
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.yaml"):
            old.unlink()
        for cell in cells:
            path = out_dir / f"{cell['spectrum_cell_id']}.yaml"
            path.write_text(_yaml_dump(cell), encoding="utf-8")
        ledger = {
            "capacity": cap.to_dict(),
            "n_cells": len(cells),
            "n_refused": len(refused),
            "refused": refused,
            "cells_per_arm_k": per,
            "expected_cells_per_arm_k": EXPECTED_CELLS_PER_ARM_K,
        }
        (out_dir / "refusal_ledger.json").write_text(
            json.dumps(ledger, indent=2), encoding="utf-8"
        )
        (out_dir / "index.json").write_text(
            json.dumps(
                {
                    "n_cells": len(cells),
                    "k_axis": list(k_axis),
                    "cells_per_arm_k": per,
                    "cells": [c["spectrum_cell_id"] for c in cells],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(
        f"fullgrid cells={len(cells)} refused={len(refused)} "
        f"k_axis={k_axis} per_arm_k={per}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
