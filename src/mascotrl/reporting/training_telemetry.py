"""Per-update training telemetry linked to spectrum/eq cell artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


_TRAINING_KEYS = (
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_frac",
    "epochs_run",
    "policy_grad_norm",
    "value_grad_norm",
    "grad_norm",
    "log_std_mean",
    "log_std_min",
    "log_std_max",
    "action_l1",
    "turnover_mean",
    "exec_turnover",
    "exec_weight_l1",
    "proj_gap",
    "proj_penalty",
    "risk_loss",
    "teamtr_policy_loss",
    "teamtr_value_loss",
    "teamtr_skips",
    "cmdp_slack_mean",
    "cmdp_lambda",
    "cmdp_limit_d",
    "cmdp_j_c_violation_frac",
    "cvar_eta",
    "cvar_nu",
    "cvar_beta",
    "trajectory_cvar",
    "cvar_violation",
    "weight_entropy",
    "rl_backend",
    "optimizer_steps",
    "loss",
    "loss_source",
    "explained_variance",
    "fps",
    "omnisafe_lambda",
    "omnisafe_ep_cost",
    "omnisafe_algo",
    "cvar_zeta",
    "reward_return_term",
    "reward_cost_term",
    "reward_turnover_penalty",
    "reward_cvar_term",
    "reward_entropy_bonus",
    "reward_composite_total",
    "mean_reward",
)


def alias_grad_norm(stats: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize ``grad_norm`` -> ``policy_grad_norm`` for telemetry consistency."""
    out = dict(stats)
    if "policy_grad_norm" not in out and "grad_norm" in out:
        try:
            out["policy_grad_norm"] = float(out["grad_norm"])
        except (TypeError, ValueError):
            pass
    return out


def reward_decomp_from_step_info(
    info: Mapping[str, Any],
    *,
    train_reward: float,
    entropy_bonus: float = 0.0,
    cvar_term: float = 0.0,
    turnover_penalty_coef: float = 0.0,
) -> dict[str, float]:
    """Per-step reward decomposition from env ``info`` + shaped train reward."""
    gross = float(info.get("gross", 0.0) or 0.0)
    cost = float(info.get("cost", 0.0) or 0.0)
    turnover = float(info.get("turnover", 0.0) or 0.0)
    turnover_pen = -abs(float(turnover_penalty_coef)) * turnover
    return {
        "reward_return_term": gross,
        "reward_cost_term": -abs(cost),
        "reward_turnover_penalty": float(turnover_pen),
        "reward_cvar_term": float(cvar_term),
        "reward_entropy_bonus": float(entropy_bonus),
        "reward_composite_total": float(train_reward),
    }


def mean_reward_decomp(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Average reward-decomposition rows into a single telemetry dict."""
    keys = (
        "reward_return_term",
        "reward_cost_term",
        "reward_turnover_penalty",
        "reward_cvar_term",
        "reward_entropy_bonus",
        "reward_composite_total",
    )
    if not rows:
        return {k: float("nan") for k in keys}
    out: dict[str, float] = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r]
        out[k] = float(sum(vals) / len(vals)) if vals else float("nan")
    return out


def normalize_training_row(
    stats: Mapping[str, Any],
    *,
    cell_id: str = "",
    update_idx: int | None = None,
    fold_id: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Extract a stable telemetry row from trainer update stats."""
    row: dict[str, Any] = {
        "cell_id": str(cell_id),
        "update_idx": update_idx,
        "fold_id": fold_id,
        "seed": seed,
    }
    for key in _TRAINING_KEYS:
        if key not in stats:
            continue
        val = stats[key]
        if isinstance(val, bool):
            row[key] = float(val)
        elif isinstance(val, (int, float)):
            row[key] = float(val)
        elif isinstance(val, str) and key in ("rl_backend", "loss_source"):
            row[key] = val
    return row


def write_training_jsonl(
    path: Path | str,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), default=str) + "\n")
    return path


def training_rows_from_diagnostics(
    diagnostics: Mapping[str, Any] | None,
    *,
    cell_id: str = "",
) -> list[dict[str, Any]]:
    """Flatten nested learning_curve / per-update lists when present."""
    if not diagnostics:
        return []
    rows: list[dict[str, Any]] = []
    curve = diagnostics.get("learning_curve") or diagnostics.get("updates")
    if isinstance(curve, list):
        for i, entry in enumerate(curve):
            if isinstance(entry, dict):
                rows.append(
                    normalize_training_row(entry, cell_id=cell_id, update_idx=i)
                )
    elif isinstance(diagnostics, dict):
        rows.append(normalize_training_row(diagnostics, cell_id=cell_id))
    return rows
