"""Desk-org companion artifact: joint portfolio + HAPPO coordination proxies.

Interpretation only. Never feeds capital gates or (by default) codename clustering.
Does not invent per-agent archetypes; joint book only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DESKORG_SCHEMA = "deskorg_v1"


def build_deskorg_artifact(
    *,
    cell_id: str,
    claim_tier: str = "narrative",
    behaviour_path: str | None = None,
    decision_trace_path: str | None = None,
    interpretability_path: str | None = None,
    coordination_proxies: Mapping[str, Any] | None = None,
    projection_slacks: Mapping[str, Any] | None = None,
    learning_curve_glob: str | None = None,
    training_telemetry_path: str | None = None,
    eval_protocol: str | None = None,
) -> dict[str, Any]:
    """Package joint-book paths + trainer coordination proxies for Ch.9."""
    coord = dict(coordination_proxies or {})
    coord.setdefault("policy_loss_last_agent_only", True)
    slacks = dict(projection_slacks or {})
    slacks.setdefault("s_delta_mean", None)
    slacks.setdefault("s_turnover_mean", None)
    slacks.setdefault(
        "note",
        "null until eval calls return_slacks=True and aggregates are present",
    )
    return {
        "schema": DESKORG_SCHEMA,
        "cell_id": str(cell_id),
        "claim_tier": str(claim_tier or "narrative"),
        "feeds_capital_gates": False,
        "feeds_codename_clustering": False,
        "claim_language": "joint_portfolio_coordination_only",
        "joint_portfolio": {
            "behaviour_path": behaviour_path,
            "decision_trace_path": decision_trace_path,
        },
        "coordination_proxies": coord,
        "projection_slacks": slacks,
        "interpretability_path": interpretability_path,
        "provenance": {
            "algo": "happo",
            "eval_protocol": eval_protocol or "combinatorial_purged_cv",
            "learning_curve_glob": learning_curve_glob,
            "training_telemetry_path": training_telemetry_path,
        },
    }


def projection_slacks_from_path0(path0: Mapping[str, Any] | None) -> dict[str, Any]:
    """Mean QP slacks from path-0 aux rows when present."""
    out: dict[str, Any] = {
        "s_delta_mean": None,
        "s_turnover_mean": None,
        "binding_frac_turnover_slack": None,
    }
    if not isinstance(path0, Mapping):
        return out
    # Prefer pre-aggregated keys; else scan nested records if ever attached.
    for key, dest in (("s_delta", "s_delta_mean"), ("s_turn", "s_turnover_mean")):
        series = path0.get(key) or path0.get(f"{key}_mean")
        if series is None:
            continue
        arr = np.asarray(series, dtype=np.float64).reshape(-1)
        if arr.size:
            out[dest] = float(np.nanmean(arr))
    return out


def write_deskorg(path: Path | str, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path
