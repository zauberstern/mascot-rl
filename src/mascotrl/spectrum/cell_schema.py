"""Declarative schema validation for spectrum cell YAML configs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from mascotrl.spectrum.protocol_tiers import PROTOCOL_TIERS
from mascotrl.spectrum.registry import PORTFOLIO_ARM_IDS

ALLOWED_WEIGHT_HEADS = frozenset(
    {
        "softmax",
        "tanh_l1",
        "sparse_tilt",
        "sparse_tilt_tsallis",
        "entmax_15",
        "dirichlet_tilt",
        "dirichlet_mean",
        "dirichlet_entropy",
        "raw",
        "discrete",
    }
)

ALIAS_PAIRS = (
    ("algo", "policy_algo"),
    ("architecture", "temporal_backend"),
    ("train_world", "train_distribution"),
)

DATE_TYPES = (str, date, datetime)


@dataclass(frozen=True)
class FieldSpec:
    typ: type
    required: bool = False
    allowed: frozenset[str] | None = None


# Flat spectrum cell keys (fullgrid + cherrypick extras).
SCHEMA: dict[str, FieldSpec] = {
    "claim_tier": FieldSpec(str),
    "cost_in_decision": FieldSpec(bool),
    "reward_shaping_ablation": FieldSpec(bool),
    "primary_train": FieldSpec(str),
    "projection_mode": FieldSpec(str),
    "turnover_limit": FieldSpec(float),
    "execution_impact_coef": FieldSpec(float),
    "rebalance_cadence": FieldSpec(str),
    "equity_bps": FieldSpec(float),
    "headline_fill": FieldSpec(str),
    "om_touch_enabled": FieldSpec(bool),
    "hedge_leg_spread_bps": FieldSpec(float),
    "execution_spread_bps": FieldSpec(float),
    "lr": FieldSpec(float),
    "use_surface_signals": FieldSpec(bool),
    "surface_obs_lane": FieldSpec(str),
    "obs_pack_path": FieldSpec(str),
    "signal_allowlist_path": FieldSpec(str),
    "lake_root": FieldSpec(str),
    "selection_start": FieldSpec(DATE_TYPES),
    "selection_end": FieldSpec(DATE_TYPES),
    "oos_start": FieldSpec(DATE_TYPES),
    "oos_end": FieldSpec(DATE_TYPES),
    "scr_mix": FieldSpec((str, bool)),
    "weight_head_tilt_gain": FieldSpec(float),
    "weight_head_temperature": FieldSpec(float),
    "entropy_coef": FieldSpec(float),
    "actor_final_gain": FieldSpec(float),
    "clip_eps": FieldSpec(float),
    "ppo_hidden": FieldSpec(int),
    "grid_kind": FieldSpec(str),
    "spectrum_cell_id": FieldSpec(str, required=True),
    "portfolio_arm": FieldSpec(str, required=True, allowed=frozenset(PORTFOLIO_ARM_IDS)),
    "n_assets": FieldSpec(int, required=True),
    "algo": FieldSpec(str, required=True),
    "policy_algo": FieldSpec(str, required=True),
    "architecture": FieldSpec(str, required=True),
    "temporal_backend": FieldSpec(str, required=True),
    "weight_head": FieldSpec(str, required=True, allowed=ALLOWED_WEIGHT_HEADS),
    "head_axis_id": FieldSpec(str, required=True),
    "objective": FieldSpec(str, required=True),
    "objective_primary": FieldSpec(bool),
    "claim_label_stem": FieldSpec(str),
    "train_world": FieldSpec(str, required=True),
    "train_distribution": FieldSpec(str, required=True),
    "policy_mode": FieldSpec(str, required=True),
    "agent": FieldSpec(str, required=True),
    "policy": FieldSpec(str, required=True),
    "action_law": FieldSpec(str, required=True),
    "protocol_tier": FieldSpec(str, required=True, allowed=frozenset(PROTOCOL_TIERS)),
    "seeds": FieldSpec(list, required=True),
    "train_env_steps": FieldSpec(int, required=True),
    "train_epochs": FieldSpec(int),
    "train_updates_per_fold": FieldSpec(int),
    "n_minibatches": FieldSpec(int),
    "cpcv_n_splits": FieldSpec(int, required=True),
    "cpcv_n_test_groups": FieldSpec(int, required=True),
    "cpcv_purge_days": FieldSpec(int, required=True),
    "cpcv_embargo_days": FieldSpec(int, required=True),
    "use_purgedcv": FieldSpec(bool),
    "bootstrap_backend": FieldSpec(str, allowed=frozenset({"custom", "arch"})),
    "require_fresh_quotes": FieldSpec(bool),
    "use_harl": FieldSpec(bool),
    "omnisafe_algo": FieldSpec(str),
    "narrative_role": FieldSpec(str),
    "universe_arm": FieldSpec(str),
    "use_equity_feature_cube": FieldSpec(bool),
    "cube_auto_enabled": FieldSpec(bool),
    "use_feature_net_extras": FieldSpec(bool),
    "hybrid_pretrain_world": FieldSpec(str),
    # Andersen QE / QE-M / full_truncation (C++ worlds.cpp); default qe_martingale.
    "heston_scheme": FieldSpec(str),
    "heston_v0": FieldSpec(float),
    "heston_theta": FieldSpec(float),
    "heston_kappa": FieldSpec(float),
    "heston_xi": FieldSpec(float),
    "heston_rho": FieldSpec(float),
    "feature_groups_exclude": FieldSpec(list),
    "feature_channels_exclude": FieldSpec(list),
    "feature_seq_len": FieldSpec(int),
    "mamba_chunk_size": FieldSpec(int),
    "requires_himem": FieldSpec(bool),
    # Proven Batch himem MiB for this cell (operator SoT for future wave submits).
    "himem_job_memory_mib": FieldSpec(int),
    "rl_backend": FieldSpec(str),
    "cvar_alpha": FieldSpec(float),
    "cvar_k_ratio": FieldSpec(float),
    "nu_lr": FieldSpec(float),
    "nu_delay": FieldSpec(float),
    "reward_weights": FieldSpec(dict),
    "cmdp": FieldSpec(dict),
    # HAPPO screening honesty stamp (fullgrid / cherrypick generators).
    "happo_dispatch_only": FieldSpec(bool),
    "dispatch_only": FieldSpec(bool),
    "happo_full_budget": FieldSpec(bool),
}


def _type_ok(val: Any, spec: FieldSpec) -> bool:
    if isinstance(spec.typ, tuple):
        return isinstance(val, spec.typ)
    return isinstance(val, spec.typ)


def validate_cell_cfg(cfg: dict[str, Any], *, path: str = "") -> None:
    """Raise ValueError on unknown keys, type errors, alias drift, or bad enums."""
    unknown = [k for k in cfg if k not in SCHEMA and not k.startswith("_")]
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}")
    for key, spec in SCHEMA.items():
        if key not in cfg:
            if spec.required:
                raise ValueError(f"{path}: missing required key {key!r}")
            continue
        val = cfg[key]
        if not _type_ok(val, spec):
            raise ValueError(f"{path}: {key!r} type {type(val).__name__} != {spec.typ}")
        if spec.allowed is not None and str(val) not in spec.allowed:
            raise ValueError(f"{path}: {key!r}={val!r} not in {sorted(spec.allowed)}")
        for a, b in ALIAS_PAIRS:
            if a in cfg and b in cfg and str(cfg[a]) != str(cfg[b]):
                raise ValueError(f"{path}: alias mismatch {a}={cfg[a]!r} vs {b}={cfg[b]!r}")
