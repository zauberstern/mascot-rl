#!/usr/bin/env python3
"""Generate the Phase 5 spectrum cherry-pick panel.

Single source of truth for ``config/spectrum/cherrypick/``. Never reads from
or writes into ``config/spectrum/fullgrid/`` (read-only source of Tier 1
sweep A-F cells). See ``SPECTRUM_CHERRYPICK.md`` for the design rationale.

Tier 1 (screening): sweeps A-F are byte-identical copies of existing
fullgrid cell YAMLs (never mutated). Sweeps G/H/I are new cells cloned from
an arm's reference cell (``{arm}_K100_single_ppo_mlp_softmax_mean_std_cao``)
with explicit key overrides.

Tier 2 (narrative): five pre-registered per-arm configs, cloned from their
fullgrid source cell with the narrative protocol tier stamped on top.

Tier 3 (optional, ``--tier3``): K=200 A-F sweeps only.

ID mismatch handled (see manifest ``id_mismatches_resolved``): sweep C's
plan text lists a third ppo head (``dirichlet_mean``). That head is not in
``ALGO_HEADS`` for ppo in ``src.spectrum.registry``
(ppo heads are ``softmax``, ``tanh_l1``, ``dirichlet_tilt``), so no fullgrid
cell exists. Phase 2 of the plan says: leave the fullgrid generator
unchanged and handle head choice explicitly in the cherry-pick generator.
Sweep C therefore copies the two fullgrid heads and clones a third cell
from the ppo reference with ``head=dirichlet_mean`` (``validate_cfg`` accepts
it). Post-RL-audit also clones ``cppo`` (Sweep A/I) and ``sdr_composite``
(Sweep D). Tier 1 A-F = 111; Tier 1 total = 147; Tier 2 narrative = 24.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.spectrum.cell_schema import validate_cell_cfg
from src.spectrum.protocol_tiers import apply_protocol_tier
from src.spectrum.registry import PORTFOLIO_ARM_IDS, validate_cfg
from src.spectrum.yaml_loader import load_cell_yaml

FULLGRID_DIR = ROOT / "config" / "spectrum" / "fullgrid"
OUT_DIR = ROOT / "config" / "spectrum" / "cherrypick"
NARRATIVE_DIR = OUT_DIR / "narrative"
K200_DIR = OUT_DIR / "k200"

ARMS: tuple[str, ...] = PORTFOLIO_ARM_IDS  # ("opt", "eq", "mix")

HOURS_PER_CELL = 1.91
USD_PER_VCPU_HOUR = 0.022
USD_PER_CELL = HOURS_PER_CELL * USD_PER_VCPU_HOUR  # 0.04202

Sweep = tuple[str, str, "str | None", str]  # (algo, body, head_or_none, objective)


def _cell_id(*, arm: str, k: int, algo: str, body: str, head: str | None, objective: str) -> str:
    agent = "multi" if algo == "happo" else "single"
    if algo in ("dqn", "happo"):
        return f"{arm}_K{k}_{agent}_{algo}_{body}_{objective}"
    if head is None:
        raise ValueError(f"algo={algo!r} requires an explicit head")
    return f"{arm}_K{k}_{agent}_{algo}_{body}_{head}_{objective}"


# Sweep A: algo sweep at each algo's admissible reference. body=mlp,
# head=softmax (first entry of _ALGO_HEADS for every continuous algo);
# objective=mean_std_cao for episode-return algos (ppo, cppo, happo), else
# mtm_pnl for the dense-reward-only algos (registry D.2 refusal).
# cppo is cloned from the ppo reference (fullgrid may lag the registry).
SWEEP_A: tuple[Sweep, ...] = (
    ("ppo", "mlp", "softmax", "mean_std_cao"),
    ("cppo", "mlp", "softmax", "mean_std_cao"),
    ("sac", "mlp", "softmax", "mtm_pnl"),
    ("td3", "mlp", "softmax", "mtm_pnl"),
    ("ddpg", "mlp", "softmax", "mtm_pnl"),
    ("dqn", "mlp", None, "mtm_pnl"),
    ("mcpg", "mlp", "softmax", "mtm_pnl"),
    ("rrl", "mlp", "softmax", "mtm_pnl"),
    ("happo", "mlp", None, "mean_std_cao"),
)
SWEEP_A_CLONE: tuple[Sweep, ...] = (
    ("cppo", "mlp", "softmax", "mean_std_cao"),
)

# Sweep B: body sweep at ppo + mean_std_cao + reference head (softmax).
# mlp already covered by sweep A.
SWEEP_B: tuple[Sweep, ...] = tuple(
    ("ppo", body, "softmax", "mean_std_cao")
    for body in ("gru", "lstm", "transformer", "mamba")
)

# Sweep C: head sweep at ppo/mlp/mean_std_cao. First two exist in fullgrid;
# dirichlet_mean is cloned (see SWEEP_C_CLONE / module docstring).
SWEEP_C: tuple[Sweep, ...] = (
    ("ppo", "mlp", "tanh_l1", "mean_std_cao"),
    ("ppo", "mlp", "dirichlet_tilt", "mean_std_cao"),
    ("ppo", "mlp", "dirichlet_mean", "mean_std_cao"),
)
SWEEP_C_FULLGRID: tuple[Sweep, ...] = SWEEP_C[:2]
SWEEP_C_CLONE: tuple[Sweep, ...] = SWEEP_C[2:]

# Sweep D: objective sweep at ppo/mlp/softmax (non-reference objectives).
# sdr_composite is cloned when absent from fullgrid.
SWEEP_D: tuple[Sweep, ...] = tuple(
    ("ppo", "mlp", "softmax", obj)
    for obj in (
        "mtm_pnl",
        "meanvar_kolm",
        "cvar_ru",
        "entropic_oce",
        "smse",
        "rsqp",
        "differential_sharpe",
        "mikkila_asym",
        "sdr_composite",
    )
)
SWEEP_D_CLONE: tuple[Sweep, ...] = (("ppo", "mlp", "softmax", "sdr_composite"),)

# Sweep E: off-policy objective cross at mlp + per-algo default head
# (tanh_l1; SPECTRUM_CHERRYPICK.md family SA-C head law).
SWEEP_E: tuple[Sweep, ...] = tuple(
    (algo, "mlp", "tanh_l1", obj)
    for algo in ("sac", "td3", "ddpg")
    for obj in ("mtm_pnl", "differential_sharpe", "mikkila_asym")
)

# Sweep F: HAPPO objective sweep (mean_std_cao already covered by sweep A).
SWEEP_F: tuple[Sweep, ...] = tuple(
    ("happo", "mlp", None, obj) for obj in ("meanvar_kolm", "cvar_ru", "entropic_oce")
)

SWEEPS_A_TO_F: dict[str, tuple[Sweep, ...]] = {
    "A": SWEEP_A,
    "B": SWEEP_B,
    "C": SWEEP_C,
    "D": SWEEP_D,
    "E": SWEEP_E,
    "F": SWEEP_F,
}

# New Tier 1 cells (G/H/I): cloned from the ppo/mlp/softmax/mean_std_cao
# reference with explicit overrides.
TRAIN_WORLDS_G: tuple[str, ...] = (
    "rbergomi",
    "gbm",
    "heston",
    "garch",
    "sabr",
    "hybrid_pretrain_finetune",
)
POLICY_MODES_H: tuple[str, ...] = (
    "archetype_carry",
    "archetype_inflation",
    "archetype_crisis",
)

# Tier 2 (narrative): (id_suffix_after_{arm}_K100_, role_label, overrides).
# overrides may set algo/objective/rl_backend for cells cloned from a base.
NARRATIVE_SPECS: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("single_ppo_mlp_softmax_mean_std_cao", "reference", {}),
    ("single_ppo_mamba_dirichlet_tilt_cvar_ru", "evidence_backed_best_sa_design", {}),
    ("multi_happo_mlp_mean_std_cao", "multi_agent_spine", {}),
    ("single_ddpg_mlp_tanh_l1_mtm_pnl", "off_policy_ls_contrast", {}),
    ("single_dqn_mlp_mtm_pnl", "discrete_foil", {}),
    (
        "single_cppo_mlp_softmax_mean_std_cao",
        "cvar_constrained_ppo",
        {"_clone_from": "single_ppo_mlp_softmax_mean_std_cao", "algo": "cppo", "policy_algo": "cppo"},
    ),
    (
        "single_ppo_mlp_softmax_sdr_composite",
        "sdr_composite_reward",
        {
            "_clone_from": "single_ppo_mlp_softmax_mean_std_cao",
            "objective": "sdr_composite",
        },
    ),
    (
        "single_ppo_mlp_softmax_mean_std_cao_rb-sb3",
        "sb3_backend_reference",
        {
            "_clone_from": "single_ppo_mlp_softmax_mean_std_cao",
            "rl_backend": "sb3",
        },
    ),
)

ID_MISMATCHES_RESOLVED: tuple[dict[str, str], ...] = (
    {
        "sweep": "C",
        "issue": (
            "Plan text lists 3 ppo heads (tanh_l1, dirichlet_tilt, "
            "dirichlet_mean) at mlp/mean_std_cao. dirichlet_mean is not in "
            "ppo's _ALGO_HEADS entry in the fullgrid generator "
            "(softmax, tanh_l1, dirichlet_tilt), so no fullgrid cell exists."
        ),
        "resolution": (
            "Leave fullgrid generator unchanged. Cherry-pick generator "
            "copies the two fullgrid heads and clones a new cell from the "
            "ppo reference with head=dirichlet_mean (validate_cfg accepts)."
        ),
    },
    {
        "sweep": "A/D/narrative",
        "issue": (
            "Post-RL-audit registry adds cppo + sdr_composite; fullgrid "
            "YAML tree may lag until regenerate. rl_backend=sb3 is a "
            "runtime routing switch, not a fullgrid axis cell."
        ),
        "resolution": (
            "Cherry-pick clones cppo (Sweep A + I + narrative), "
            "sdr_composite (Sweep D + narrative), and rb-sb3 narrative "
            "cells from the ppo reference. Tier 1 = 147; Tier 2 = 24."
        ),
    },
)


def _load_yaml(path: Path) -> dict[str, Any]:
    return load_cell_yaml(path)


def _yaml_dump(cfg: dict[str, Any], header: list[str]) -> str:
    lines = list(header)
    for key, val in cfg.items():
        if key.startswith("_"):
            continue
        if isinstance(val, bool):
            lines.append(f"{key}: {'true' if val else 'false'}")
        elif isinstance(val, (list, tuple)):
            lines.append(f"{key}: [{', '.join(str(v) for v in val)}]")
        else:
            lines.append(f"{key}: {val}")
    return "\n".join(lines) + "\n"


def _write_cell(path: Path, cfg: dict[str, Any], header: list[str], *, dry_run: bool) -> None:
    validate_cell_cfg(cfg, path=str(path))
    if dry_run:
        return
    path.write_text(_yaml_dump(cfg, header), encoding="utf-8")


def _require_fullgrid_cell(cell_id: str, *, sweep_label: str, arm: str) -> Path:
    src = FULLGRID_DIR / f"{cell_id}.yaml"
    if not src.is_file():
        raise SystemExit(
            f"cherrypick sweep {sweep_label} arm={arm}: cell_id={cell_id!r} "
            "not found in fullgrid (id mismatch; see module docstring)"
        )
    return src


def _clone_cell(
    *,
    arm: str,
    k: int,
    algo: str,
    body: str,
    head: str | None,
    objective: str,
    out_dir: Path,
    dry_run: bool,
    sweep_label: str,
    extra: dict[str, Any] | None = None,
) -> str:
    """Clone the ppo reference and override algo/head/objective (+ optional keys)."""
    ref_id = _cell_id(
        arm=arm, k=k, algo="ppo", body="mlp", head="softmax", objective="mean_std_cao"
    )
    ref_cfg = _load_yaml(_require_fullgrid_cell(ref_id, sweep_label=f"{sweep_label}-ref", arm=arm))
    cell_id = _cell_id(arm=arm, k=k, algo=algo, body=body, head=head, objective=objective)
    cfg = dict(ref_cfg)
    cfg["algo"] = algo
    cfg["policy_algo"] = algo
    if head is not None:
        cfg["weight_head"] = head
        cfg["head_axis_id"] = head
        cfg["action_law"] = head
    cfg["architecture"] = body
    cfg["temporal_backend"] = body
    cfg["objective"] = objective
    cfg["spectrum_cell_id"] = cell_id
    cfg["grid_kind"] = "cherrypick"
    # Arm-faithful label stem (eq=stk_ret, opt=dh_ret_lagdelta; mix stamped at load).
    if arm == "eq":
        cfg["claim_label_stem"] = "stk_ret"
    elif arm == "opt":
        cfg["claim_label_stem"] = "dh_ret_lagdelta"
    elif arm == "mix":
        cfg.pop("claim_label_stem", None)
    # Episode-weight objectives must be primary so sample_weight reaches PPO.
    _EPISODE = {
        "mean_std_cao",
        "meanvar_kolm",
        "cvar_ru",
        "entropic_oce",
        "smse",
        "rsqp",
    }
    if objective in _EPISODE:
        cfg["objective_primary"] = True
    if algo == "happo":
        cfg["agent"] = "multi"
        cfg["policy"] = "multi_agent"
    else:
        cfg["agent"] = "single"
        cfg["policy"] = "single_agent"
    if extra:
        cfg.update({k: v for k, v in extra.items() if not str(k).startswith("_")})
    cfg.pop("body", None)
    cfg.pop("head", None)
    validate_cfg(cfg)
    _write_cell(
        out_dir / f"{cell_id}.yaml",
        cfg,
        [
            f"# Cherrypick Tier 1 sweep {sweep_label} (cloned) cell: {cell_id}",
            "# Generated by scripts/generate_cherrypick_panel.py; do not hand-edit.",
        ],
        dry_run=dry_run,
    )
    return cell_id


def _clone_head_cell(
    *,
    arm: str,
    k: int,
    algo: str,
    body: str,
    head: str,
    objective: str,
    out_dir: Path,
    dry_run: bool,
    sweep_label: str,
) -> str:
    return _clone_cell(
        arm=arm,
        k=k,
        algo=algo,
        body=body,
        head=head,
        objective=objective,
        out_dir=out_dir,
        dry_run=dry_run,
        sweep_label=sweep_label,
    )

def generate_af_sweeps(
    out_dir: Path, *, k: int, dry_run: bool
) -> tuple[dict[str, Any], list[str]]:
    """Copy sweeps A-F from fullgrid; clone SWEEP_*_CLONE cells; validate all."""
    report: dict[str, Any] = {}
    all_ids: list[str] = []
    clone_set = set(SWEEP_C_CLONE) | set(SWEEP_A_CLONE) | set(SWEEP_D_CLONE)
    for label, sweep in SWEEPS_A_TO_F.items():
        per_arm_ids: dict[str, list[str]] = {}
        for arm in ARMS:
            arm_ids: list[str] = []
            for algo, body, head, objective in sweep:
                cell_id = _cell_id(
                    arm=arm, k=k, algo=algo, body=body, head=head, objective=objective
                )
                if (algo, body, head, objective) in clone_set:
                    cell_id = _clone_cell(
                        arm=arm,
                        k=k,
                        algo=algo,
                        body=body,
                        head=head,
                        objective=objective,
                        out_dir=out_dir,
                        dry_run=dry_run,
                        sweep_label=label,
                    )
                else:
                    src = _require_fullgrid_cell(cell_id, sweep_label=label, arm=arm)
                    cfg = _load_yaml(src)
                    cfg["grid_kind"] = "cherrypick"
                    validate_cfg(cfg)
                    _write_cell(
                        out_dir / f"{cell_id}.yaml",
                        cfg,
                        [
                            f"# Cherrypick Tier 1 sweep {label} cell: {cell_id}",
                            "# Generated by scripts/generate_cherrypick_panel.py; do not hand-edit.",
                        ],
                        dry_run=dry_run,
                    )
                arm_ids.append(cell_id)
                all_ids.append(cell_id)
            per_arm_ids[arm] = arm_ids
        report[label] = {
            "per_arm": len(sweep),
            "total": len(sweep) * len(ARMS),
            "cell_ids": per_arm_ids,
        }
    return report, all_ids


def generate_sweep_g(out_dir: Path, *, k: int, dry_run: bool) -> dict[str, list[str]]:
    """Train-world sweep on the ppo reference cell. Suffix ``_tw-{world}``."""
    report: dict[str, list[str]] = {}
    for arm in ARMS:
        ref_id = _cell_id(arm=arm, k=k, algo="ppo", body="mlp", head="softmax", objective="mean_std_cao")
        ref_cfg = _load_yaml(_require_fullgrid_cell(ref_id, sweep_label="G-ref", arm=arm))
        arm_ids: list[str] = []
        for world in TRAIN_WORLDS_G:
            cell_id = f"{ref_id}_tw-{world}"
            cfg = dict(ref_cfg)
            cfg["train_world"] = world
            cfg["train_distribution"] = world
            cfg["spectrum_cell_id"] = cell_id
            cfg["grid_kind"] = "cherrypick"
            validate_cfg(cfg)
            _write_cell(
                out_dir / f"{cell_id}.yaml",
                cfg,
                [
                    f"# Cherrypick Tier 1 sweep G (train-world) cell: {cell_id}",
                    "# Generated by scripts/generate_cherrypick_panel.py; do not hand-edit.",
                ],
                dry_run=dry_run,
            )
            arm_ids.append(cell_id)
        report[arm] = arm_ids
    return report


def generate_sweep_h(out_dir: Path, *, k: int, dry_run: bool) -> dict[str, list[str]]:
    """Policy-mode sweep on the ppo reference cell. Suffix ``_pm-{mode}``."""
    report: dict[str, list[str]] = {}
    for arm in ARMS:
        ref_id = _cell_id(arm=arm, k=k, algo="ppo", body="mlp", head="softmax", objective="mean_std_cao")
        ref_cfg = _load_yaml(_require_fullgrid_cell(ref_id, sweep_label="H-ref", arm=arm))
        arm_ids: list[str] = []
        for mode in POLICY_MODES_H:
            cell_id = f"{ref_id}_pm-{mode}"
            cfg = dict(ref_cfg)
            cfg["policy_mode"] = mode
            cfg["spectrum_cell_id"] = cell_id
            cfg["grid_kind"] = "cherrypick"
            validate_cfg(cfg)
            _write_cell(
                out_dir / f"{cell_id}.yaml",
                cfg,
                [
                    f"# Cherrypick Tier 1 sweep H (policy-mode) cell: {cell_id}",
                    "# Generated by scripts/generate_cherrypick_panel.py; do not hand-edit.",
                ],
                dry_run=dry_run,
            )
            arm_ids.append(cell_id)
        report[arm] = arm_ids
    return report


def generate_sweep_i(out_dir: Path, *, k: int, dry_run: bool) -> list[str]:
    """Crucible foil (eq only): sweep A cells with universe_arm=dyn_crucible."""
    arm = "eq"
    ids: list[str] = []
    clone_set = set(SWEEP_A_CLONE)
    for algo, body, head, objective in SWEEP_A:
        base_id = _cell_id(arm=arm, k=k, algo=algo, body=body, head=head, objective=objective)
        local = out_dir / f"{base_id}.yaml"
        if local.is_file():
            cfg = _load_yaml(local)
        elif (algo, body, head, objective) in clone_set:
            _clone_cell(
                arm=arm,
                k=k,
                algo=algo,
                body=body,
                head=head,
                objective=objective,
                out_dir=out_dir,
                dry_run=dry_run,
                sweep_label="I-base",
            )
            cfg = _load_yaml(out_dir / f"{base_id}.yaml")
        else:
            cfg = _load_yaml(_require_fullgrid_cell(base_id, sweep_label="I-base", arm=arm))
        cell_id = f"{base_id}_uni-crucible"
        cfg = dict(cfg)
        cfg["universe_arm"] = "dyn_crucible"
        cfg["spectrum_cell_id"] = cell_id
        cfg["grid_kind"] = "cherrypick"
        validate_cfg(cfg)
        _write_cell(
            out_dir / f"{cell_id}.yaml",
            cfg,
            [
                f"# Cherrypick Tier 1 sweep I (crucible foil) cell: {cell_id}",
                "# Generated by scripts/generate_cherrypick_panel.py; do not hand-edit.",
            ],
            dry_run=dry_run,
        )
        ids.append(cell_id)
    return ids


def generate_tier1(out_dir: Path, *, dry_run: bool) -> dict[str, Any]:
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.yaml"):
            old.unlink()
    af_report, af_ids = generate_af_sweeps(out_dir, k=100, dry_run=dry_run)
    g_report = generate_sweep_g(out_dir, k=100, dry_run=dry_run)
    h_report = generate_sweep_h(out_dir, k=100, dry_run=dry_run)
    i_ids = generate_sweep_i(out_dir, k=100, dry_run=dry_run)

    all_ids = list(af_ids)
    for arm_ids in g_report.values():
        all_ids.extend(arm_ids)
    for arm_ids in h_report.values():
        all_ids.extend(arm_ids)
    all_ids.extend(i_ids)

    if len(set(all_ids)) != len(all_ids):
        raise SystemExit("cherrypick tier1 produced duplicate cell ids")

    return {
        "n_cells": len(all_ids),
        "sweeps_a_to_f": af_report,
        "sweep_g_train_world": {"per_arm": len(TRAIN_WORLDS_G), "total": len(TRAIN_WORLDS_G) * len(ARMS), "cell_ids": g_report},
        "sweep_h_policy_mode": {"per_arm": len(POLICY_MODES_H), "total": len(POLICY_MODES_H) * len(ARMS), "cell_ids": h_report},
        "sweep_i_crucible_foil": {"total": len(i_ids), "cell_ids": i_ids},
        "cell_ids": all_ids,
    }


def generate_narrative(out_dir: Path, *, dry_run: bool) -> dict[str, Any]:
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.yaml"):
            old.unlink()
    report: dict[str, list[str]] = {}
    all_ids: list[str] = []
    for arm in ARMS:
        arm_ids: list[str] = []
        for suffix, role, overrides in NARRATIVE_SPECS:
            cell_id = f"{arm}_K100_{suffix}"
            clone_from = overrides.get("_clone_from")
            if clone_from:
                base_id = f"{arm}_K100_{clone_from}"
                cfg = _load_yaml(_require_fullgrid_cell(base_id, sweep_label="narrative", arm=arm))
                for key, val in overrides.items():
                    if str(key).startswith("_"):
                        continue
                    cfg[key] = val
                cfg["spectrum_cell_id"] = cell_id
            else:
                cfg = _load_yaml(_require_fullgrid_cell(cell_id, sweep_label="narrative", arm=arm))
            apply_protocol_tier(cfg, "narrative")
            cfg["claim_tier"] = "narrative"
            cfg["grid_kind"] = "cherrypick_narrative"
            cfg["narrative_role"] = role
            # Narrative HAPPO is full-budget; drop screening dispatch stamp if copied.
            cfg.pop("happo_dispatch_only", None)
            cfg.pop("dispatch_only", None)
            validate_cfg(cfg)
            _write_cell(
                out_dir / f"{cell_id}.yaml",
                cfg,
                [
                    f"# Cherrypick Tier 2 narrative cell: {cell_id} ({role})",
                    "# Generated by scripts/generate_cherrypick_panel.py; do not hand-edit.",
                ],
                dry_run=dry_run,
            )
            arm_ids.append(cell_id)
            all_ids.append(cell_id)
        report[arm] = arm_ids
    return {"n_cells": len(all_ids), "cell_ids": report}


def generate_tier3(out_dir: Path, *, dry_run: bool) -> dict[str, Any]:
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.yaml"):
            old.unlink()
    af_report, af_ids = generate_af_sweeps(out_dir, k=200, dry_run=dry_run)
    return {"n_cells": len(af_ids), "sweeps_a_to_f": af_report, "cell_ids": af_ids}


def _cost_block(n_cells: int) -> dict[str, float]:
    return {
        "n_cells": n_cells,
        "wall_hours": round(n_cells * HOURS_PER_CELL, 4),
        "usd_at_1_vcpu": round(n_cells * USD_PER_CELL, 4),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--tier3", action="store_true", help="Also emit K=200 A-F sweeps into cherrypick/k200/")
    args = p.parse_args(argv)
    dry_run = bool(args.dry_run)

    if not dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    tier1 = generate_tier1(OUT_DIR, dry_run=dry_run)
    narrative = generate_narrative(NARRATIVE_DIR, dry_run=dry_run)
    tier3 = generate_tier3(K200_DIR, dry_run=dry_run) if args.tier3 else None

    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cost_law": {
            "formula": "usd = n_cells * 1.91h * $0.022/vcpu-hour",
            "hours_per_cell": HOURS_PER_CELL,
            "usd_per_vcpu_hour": USD_PER_VCPU_HOUR,
            "usd_per_cell": round(USD_PER_CELL, 5),
        },
        "id_mismatches_resolved": list(ID_MISMATCHES_RESOLVED),
        "arms": list(ARMS),
        "tiers": {
            "screening": {**tier1, "cost": _cost_block(tier1["n_cells"])},
            "narrative": {**narrative, "cost": _cost_block(narrative["n_cells"])},
        },
        "unexpected_refusals": [],
    }
    if tier3 is not None:
        manifest["tiers"]["k200"] = {**tier3, "cost": _cost_block(tier3["n_cells"])}

    total_cells = tier1["n_cells"] + narrative["n_cells"] + (tier3["n_cells"] if tier3 else 0)
    manifest["set_algebra"] = {
        "screening_n": tier1["n_cells"],
        "narrative_n": narrative["n_cells"],
        "k200_n": tier3["n_cells"] if tier3 else 0,
        "total_n": total_cells,
        "screening_narrative_overlap": sorted(
            set(tier1["cell_ids"]) & {cid for ids in narrative["cell_ids"].values() for cid in ids}
        ),
    }

    if not dry_run:
        (OUT_DIR / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )

    print(
        f"cherrypick screening={tier1['n_cells']} narrative={narrative['n_cells']} "
        f"k200={tier3['n_cells'] if tier3 else 'skipped'} total={total_cells}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
