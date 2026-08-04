"""YAML honesty: every key in eq_alloc spine must be read or explicitly unused."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

# Keys that are provenance / documentation only and intentionally unread
# by the research-alpha CPCV path. Spine-only keys that ARE now wired into
# train_research_hist / PPO must NOT appear here.
KNOWN_UNUSED_EQ_ALLOC: frozenset[str] = frozenset(
    {
        # Option-surface tensor geometry (eq arm has no strikes/maturities).
        "n_strikes",
        "n_maturities",
        # Spine CMDP feature extractor dims (Arm B / happo path only).
        "d_model",
        "d_state",
        "macro_dim",
        "seq_len",
        "temporal_backend",
        "share_temporal_encoder",
        "actor_backend",
        "use_dhgnn",
        "spatial_mode",
        "use_projection",
        "actor_portfolio_state",
        "n_paths",
        "n_steps",
        # Campaign uses CLI seeds; YAML eval_seeds is advisory.
        "eval_seeds",
        # Provenance.
        "run_label",
        "run_notes",
        # Hygiene stamps.
        "reward_shaping_ablation",
        # cost_in_decision is enforced in research_alpha_train (B-COST); not unused.
        "spa_happo_as_claimant",
        "run_benchmark_panel",
        # Calendar mirrors equity_panel constants when campaign hardcodes.
        "hist_panel_start",
        "is_hist_start",
        "is_hist_end",
        "oos_start",
        "oos_end",
        # Router fallback when arm.id present.
        "portfolio_arm",
        "train_world",
        "train_label_stem",
        "forbid_atm_equity_proxy",
        "universe_mode",
        # Read only by the separate scripts/train_happo.py CMDP spine entry
        # point (config/workflows/happo_cmdp_mamba_k50.yaml), not by
        # scripts/run_eq_alloc_campaign.py's historical_arm_env research
        # path: the campaign only ever *writes* cfg["n_assets"] (derived
        # from the realized post-coverage-filter DII selection, not the
        # YAML value) and never routes through should_route_eq_via_cmdp.
        "n_assets",
        "route_eq_via_cmdp",
    }
)

# Keys the research / campaign path is expected to read.
RESEARCH_READ_KEYS: frozenset[str] = frozenset(
    {
        "lr",
        "entropy_coef",
        "train_epochs",
        "n_minibatches",
        "train_episodes",
        "train_env_steps",
        "min_optimizer_steps",
        "min_optimizer_steps_total",
        "warm_start_folds",
        "weight_head",
        "weight_head_tilt_gain",
        "weight_head_temperature",
        "cost_in_decision",
        "scr_mix",
        "scr_beta",
        "ppo_hidden",
        "gamma",
        "gae_lambda",
        "turnover_limit",
        "claim_label_stem",
        "use_equity_feature_cube",
        "feature_seq_len",
        "include_residual_momentum",
        "equity_bps",
        "impact_c_eq",
        "headline_fill",
        "primary_train",
        "policy",
        "projection_mode",
        "rebalance_cadence",
        "arm",
        "cpcv_n_splits",
        "cpcv_n_test_groups",
        "cpcv_purge_days",
        "cpcv_embargo_days",
        "use_surface_signals",
        "signal_allowlist_path",
        "surface_obs_lane",
        "obs_pack_path",
        "actor_final_gain",
        "clip_eps",
        "ppo_hidden",
        "weight_head_temperature",
        "train_updates_per_fold",
        # A1: YAML-authoritative when CLI --universe-arm omitted.
        "universe_arm",
        # A16: CLI --max-pool default None resolves from YAML.
        "max_pool",
        # A4: intra-fold torch checkpoint cadence.
        "checkpoint_every_n_episodes",
        # H1.1b: keep full 2014-2024 OOS panel with (T,K) availability mask.
        "use_availability_mask",
        # CLI --k overrides YAML k; campaign always touches cfg['k'] for honesty.
        "k",
        # CRUCIBLE sleeve universe (eq_alloc_crucible_k100.yaml).
        "crucible",
        "universe_cadence",
        "policy_cadence",
        "feature_extras",
        "feature_groups_exclude",
        "feature_channels_exclude",
        "rl_backend",
        "cvar_alpha",
        "cvar_k_ratio",
        "nu_lr",
        "nu_delay",
        "reward_weights",
        "cmdp",
        # Friction / om_touch plugin block (read by friction_spec_from_cfg).
        "plugins",
    }
)


class TrackingDict(dict):
    """dict subclass that records every key actually read via ``__getitem__``,
    ``get`` or ``in``.

    A11: ``RESEARCH_READ_KEYS`` was a hand-maintained list documenting what
    *should* be read; it never verified what the running code *actually*
    reads. Wrap a loaded workflow config in ``TrackingDict`` at the campaign
    entry point and call :func:`assert_yaml_honesty_tracked` at the end of
    the run so a key that looks wired but is never touched at runtime fails
    the gate.

    ``dict(cfg)`` copies made deeper in the call graph (e.g.
    ``cfg_local = dict(cfg)``) would normally lose the tracking wrapper;
    :meth:`copy` and the ``_accessed`` passthrough constructor keep the
    same underlying accessed-set alive across those copies so the gate
    reflects the whole run, not just the outermost frame.
    """

    def __init__(self, *args: Any, _accessed: set[str] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._accessed: set[str] = _accessed if _accessed is not None else set()

    def __getitem__(self, key: Any) -> Any:
        self._accessed.add(key)
        return super().__getitem__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        self._accessed.add(key)
        return super().get(key, default)

    def __contains__(self, key: Any) -> bool:
        self._accessed.add(key)
        return super().__contains__(key)

    def copy(self) -> "TrackingDict":
        return TrackingDict(self, _accessed=self._accessed)

    @property
    def accessed_keys(self) -> frozenset[str]:
        return frozenset(self._accessed)


def track_copy(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """``dict(cfg)`` that preserves ``TrackingDict`` tracking when present.

    Use in place of a bare ``dict(cfg)`` on any code path reachable from a
    campaign entry point wrapped in :class:`TrackingDict`, so a value read
    only from the copy still counts as accessed.
    """
    if isinstance(cfg, TrackingDict):
        return TrackingDict(cfg, _accessed=cfg._accessed)
    return dict(cfg)


def assert_yaml_honesty_tracked(
    tracking: "TrackingDict",
    top_level_keys: set[str] | frozenset[str],
    *,
    known_unused: set[str] | frozenset[str] | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Fail if any top-level key was neither actually accessed nor known-unused.

    Unlike :func:`assert_yaml_honesty`, this checks real runtime access
    recorded on ``tracking`` rather than a hand-maintained "should read" list.
    """
    unused = set(known_unused if known_unused is not None else KNOWN_UNUSED_EQ_ALLOC)
    accessed = set(tracking.accessed_keys)
    orphans = sorted(set(top_level_keys) - unused - accessed)
    if orphans:
        raise AssertionError(
            f"YAML honesty (tracked) fail{f' in {path}' if path else ''}: "
            f"keys declared but never read at runtime: {orphans}. "
            "Wire them into the executed path or add to KNOWN_UNUSED_EQ_ALLOC."
        )
    return {
        "path": str(path) if path else None,
        "n_keys": len(top_level_keys),
        "n_accessed": len(set(top_level_keys) & accessed),
        "n_unused": len(set(top_level_keys) & unused),
        "orphans": orphans,
    }


def load_workflow_keys(path: str | Path) -> set[str]:
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(cfg, Mapping):
        raise TypeError(f"workflow YAML root must be a mapping: {path}")
    return set(cfg.keys())


def assert_yaml_honesty(
    path: str | Path,
    *,
    known_unused: set[str] | frozenset[str] | None = None,
    read_keys: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """Fail if any top-level key is neither read nor listed as known_unused."""
    keys = load_workflow_keys(path)
    unused = set(known_unused if known_unused is not None else KNOWN_UNUSED_EQ_ALLOC)
    read = set(read_keys if read_keys is not None else RESEARCH_READ_KEYS)
    orphans = sorted(keys - unused - read)
    if orphans:
        raise AssertionError(
            f"YAML honesty fail: unread keys in {path}: {orphans}. "
            f"Wire them into the executed path or add to known_unused."
        )
    return {
        "path": str(path),
        "n_keys": len(keys),
        "n_read": len(keys & read),
        "n_unused": len(keys & unused),
        "orphans": orphans,
    }


def assert_turnover_cap_honesty(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse advertising ``turnover_limit`` unless ``projection_mode`` is hard.

    Soft / monitor / off projection is a passthrough on the research path; a
    non-null ``turnover_limit`` under those modes is decorative and must not
    appear in a headline workflow.
    """
    mode = str(cfg.get("projection_mode") or "soft").lower()
    limit = cfg.get("turnover_limit")
    if limit is not None and mode != "hard":
        raise AssertionError(
            f"turnover_limit={limit!r} advertised under projection_mode={mode!r}; "
            "only projection_mode='hard' enforces the cap (W1.3)"
        )
    return {
        "projection_mode": mode,
        "turnover_limit": limit,
        "turnover_cap_enforced": mode == "hard" and limit is not None,
    }


def refuse_rrl_double_dsr(cfg: Mapping[str, Any]) -> None:
    """Refuse stacking RRL's internal DSR with spectrum DSR via objective or reward.

    ``RRLAgent`` already shapes rewards with Moody differential Sharpe. Setting
    the spectrum objective OR the dense reward key to the same estimand would
    double-count DSR.
    """
    algo = str(cfg.get("algo") or cfg.get("policy") or "").lower().strip()
    objective = str(cfg.get("objective") or "").lower().strip()
    reward = str(cfg.get("reward") or "").lower().strip()
    risk = cfg.get("risk") if isinstance(cfg.get("risk"), Mapping) else {}
    if not objective and isinstance(risk, Mapping):
        objective = str(risk.get("mode") or "").lower().strip()
    if algo == "rrl" and (
        objective in {"differential_sharpe", "dsr"}
        or reward in {"differential_sharpe", "dsr"}
    ):
        raise ValueError(
            "algo=rrl already applies DifferentialSharpe internally; "
            "refuse objective/reward=differential_sharpe (double-DSR)"
        )
