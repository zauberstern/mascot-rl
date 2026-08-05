#!/usr/bin/env python3
"""Generate RC6 do-or-die cherrypick YAMLs (sparse_tilt pivot panel).

Writes under config/spectrum/cherrypick/rc6/ and a canary subset under
config/spectrum/cherrypick/rc6_canary/.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from mascotrl.spectrum.cell_schema import validate_cell_cfg
from mascotrl.spectrum.registry import validate_cfg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config" / "spectrum" / "cherrypick" / "rc6"
CANARY = ROOT / "config" / "spectrum" / "cherrypick" / "rc6_canary"

RC6_OVERRIDES = {
    "claim_tier": "research",
    "train_updates_per_fold": 3,
    "rl_backend": "custom",
    "cost_in_decision": True,
    "reward_shaping_ablation": True,
    "primary_train": "historical_arm_env",
    "projection_mode": "soft",
    "execution_impact_coef": 0.5,
    "rebalance_cadence": "daily",
    "equity_bps": 5.0,
    "headline_fill": "pct75",
    "om_touch_enabled": True,
    "hedge_leg_spread_bps": 5.0,
    "execution_spread_bps": 5.0,
    "lr": 0.0003,
    "use_surface_signals": True,
    "use_equity_feature_cube": True,
    "surface_obs_lane": "geometry_lite",
    "selection_start": "2003-01-01",
    "selection_end": "2012-12-31",
    "oos_start": "2014-01-01",
    "oos_end": "2024-12-31",
    "scr_mix": False,
    "weight_head_tilt_gain": 5.0,
    "weight_head_temperature": 1.0,
    "grid_kind": "cherrypick_rc6",
    "n_assets": 100,
    "architecture": "mlp",
    "temporal_backend": "mlp",
    "train_world": "historical",
    "train_distribution": "historical",
    "policy_mode": "shared",
    "agent": "single",
    "policy": "single_agent",
    "protocol_tier": "screening",
    "seeds": [0],
    "train_env_steps": 300000,
    "cpcv_n_splits": 6,
    "cpcv_n_test_groups": 2,
    "cpcv_purge_days": 21,
    "cpcv_embargo_days": 21,
    "train_epochs": 8,
    "n_minibatches": 8,
    "entropy_coef": 0.01,
    "actor_final_gain": 0.1,
    "clip_eps": 0.3,
    "ppo_hidden": 256,
    "requires_himem": True,
}

# Sweep A: sparse_tilt PPO x 10 objectives
SWEEP_A = [
    ("ppo", "sparse_tilt", obj)
    for obj in (
        "mean_std_cao",
        "cvar_ru",
        "entropic_oce",
        "differential_sharpe",
        "mtm_pnl",
        "meanvar_kolm",
        "mikkila_asym",
        "smse",
        "rsqp",
        "sdr_composite",
    )
]

# Sweep B: sparse_tilt off-policy
SWEEP_B = [
    ("sac", "sparse_tilt", "mtm_pnl"),
    ("td3", "sparse_tilt", "mtm_pnl"),
    ("ddpg", "sparse_tilt", "mtm_pnl"),
]

# Sweep C: sparse_tilt CPPO
SWEEP_C = [("cppo", "sparse_tilt", "mean_std_cao")]

# Sweep D: softmax control
SWEEP_D = [
    ("ppo", "softmax", obj)
    for obj in ("mean_std_cao", "cvar_ru", "entropic_oce", "mtm_pnl")
]

# Sweep E: tanh_l1 control
SWEEP_E = [
    ("ppo", "tanh_l1", "mean_std_cao"),
    ("sac", "tanh_l1", "mtm_pnl"),
    ("td3", "tanh_l1", "mtm_pnl"),
    ("ddpg", "tanh_l1", "mtm_pnl"),
]

# Sweep F: DQN
SWEEP_F = [("dqn", "discrete", "mtm_pnl")]

# Sweep G: policy modes
SWEEP_G_MODES = ("archetype_carry", "archetype_crisis", "archetype_inflation")

# Sweep H: train worlds (feature cube off — RASP lock requires historical)
SWEEP_H_WORLDS = ("heston", "gbm", "hybrid_pretrain_finetune")

# Extra canary-only cells not in main sweeps
EXTRA_CANARY = [
    ("eq", "ddpg", "softmax", "mtm_pnl"),
]
# Sweep I: crucible foils
SWEEP_I = True

CANARY_STEMS = {
    "eq_K100_single_ppo_mlp_sparse_tilt_mean_std_cao",
    "eq_K100_single_ppo_mlp_sparse_tilt_cvar_ru",
    "eq_K100_single_ppo_mlp_sparse_tilt_differential_sharpe",
    "eq_K100_single_ppo_mlp_softmax_mean_std_cao",
    "eq_K100_single_ppo_mlp_softmax_cvar_ru",
    "eq_K100_single_ppo_mlp_tanh_l1_mean_std_cao",
    "eq_K100_single_ddpg_mlp_softmax_mtm_pnl",
    "eq_K100_single_dqn_mlp_discrete_mtm_pnl",
    "eq_K100_single_ppo_mlp_sparse_tilt_mean_std_cao_hardtau",
    "eq_K100_single_sac_mlp_sparse_tilt_mtm_pnl",
}


def _cell(
    arm: str,
    algo: str,
    head: str,
    objective: str,
    *,
    policy_mode: str = "shared",
    train_world: str = "historical",
    universe_foil: str | None = None,
    hard_tau: bool = False,
    suffix: str = "",
) -> dict:
    cfg = copy.deepcopy(RC6_OVERRIDES)
    stem = f"{arm}_K100_single_{algo}_mlp_{head}_{objective}"
    if policy_mode != "shared":
        stem += f"_pm-{policy_mode}"
        cfg["policy_mode"] = policy_mode
    if train_world != "historical":
        if train_world == "hybrid_pretrain_finetune":
            stem += "_hybrid_heston"
        else:
            stem += f"_tw-{train_world}"
        cfg["train_world"] = train_world
        cfg["train_distribution"] = train_world
    if universe_foil:
        stem += f"_uni-{universe_foil}"
        cfg["universe_arm"] = "dyn_crucible" if universe_foil == "crucible" else "dyn_hrp"
    if hard_tau:
        stem += "_hardtau"
        cfg["projection_mode"] = "hard"
        cfg["turnover_limit"] = 0.05
    if suffix:
        stem += suffix
    cfg["spectrum_cell_id"] = stem
    cfg["portfolio_arm"] = arm
    cfg["algo"] = algo
    cfg["policy_algo"] = algo
    cfg["weight_head"] = head
    cfg["head_axis_id"] = head
    cfg["objective"] = objective
    cfg["action_law"] = head
    if head == "tanh_l1":
        cfg["weight_head_tilt_gain"] = 1.0
    if algo == "dqn":
        cfg["architecture"] = "mlp"
        cfg["temporal_backend"] = "mlp"
        # DQN does not use clip_eps / entropy / actor gain the same way
        cfg.pop("clip_eps", None)
    if algo in ("sac", "td3", "ddpg") and head == "softmax":
        # Off-policy softmax control for canary DDPG cell
        pass
    if objective in (
        "mtm_pnl",
        "differential_sharpe",
        "mikkila_asym",
        "sdr_composite",
    ):
        # Dense-reward path; episode-weight objectives stay as-is.
        pass
    return cfg


def _all_cells_for_arm(arm: str) -> list[dict]:
    cells: list[dict] = []
    for algo, head, obj in SWEEP_A + SWEEP_B + SWEEP_C + SWEEP_D + SWEEP_E + SWEEP_F:
        cells.append(_cell(arm, algo, head, obj))
    for mode in SWEEP_G_MODES:
        cells.append(
            _cell(arm, "ppo", "sparse_tilt", "mean_std_cao", policy_mode=mode)
        )
    for tw in SWEEP_H_WORLDS:
        c = _cell(arm, "ppo", "sparse_tilt", "mean_std_cao", train_world=tw)
        # Synthetic / hybrid worlds refuse the equity feature cube (RASP lock).
        c["use_equity_feature_cube"] = False
        cells.append(c)
    if SWEEP_I and arm == "eq":
        cells.append(
            _cell(
                arm,
                "ppo",
                "sparse_tilt",
                "mean_std_cao",
                universe_foil="crucible",
            )
        )
        cells.append(
            _cell(
                arm,
                "ppo",
                "softmax",
                "mean_std_cao",
                universe_foil="crucible",
            )
        )
    # Hard-tau ablation (eq only, canary + fleet)
    if arm == "eq":
        cells.append(
            _cell(
                arm,
                "ppo",
                "sparse_tilt",
                "mean_std_cao",
                hard_tau=True,
            )
        )
        cells.append(_cell(arm, "ddpg", "softmax", "mtm_pnl"))
    return cells


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CANARY.mkdir(parents=True, exist_ok=True)
    written = 0
    canary_n = 0
    errors: list[str] = []
    for arm in ("eq", "opt", "mix"):
        for cfg in _all_cells_for_arm(arm):
            stem = cfg["spectrum_cell_id"]
            try:
                validate_cell_cfg(cfg)
                validate_cfg(cfg)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{stem}: {exc}")
                continue
            path = OUT / f"{stem}.yaml"
            with path.open("w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
            written += 1
            if stem in CANARY_STEMS:
                cpath = CANARY / f"{stem}.yaml"
                with cpath.open("w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
                canary_n += 1
    print(f"wrote {written} RC6 cells to {OUT}")
    print(f"wrote {canary_n} canary cells to {CANARY}")
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(" ", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
