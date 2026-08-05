"""Research hist train: HistoricalArmEnv + single-agent PPO + FrictionSpec."""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from mascotrl.arms import arm_spec_from_cfg
from mascotrl.env.historical_env import HistoricalArmEnv
from mascotrl.eval.differential_sharpe import DifferentialSharpe
from mascotrl.eval.friction import assert_friction_parity
from mascotrl.eval.residualization import fit_ff4_residualizer, freeze_residualizer
from mascotrl.eval.yaml_honesty import track_copy
from mascotrl.policy.objective_factory import (
    episode_weights,
    mikkila_asym_reward,
    objective_gradient_path_for,
    resolve_objective_mode,
    sdr_composite_reward,
)
from mascotrl.policy.single_agent import make_single_agent
from mascotrl.spectrum.registry import validate_cfg
from mascotrl.reporting.research_alpha_router import (
    RESEARCH_PRIMARY_ALLOWED,
    RESEARCH_PRIMARY_HIST,
    research_train_friction_pair,
    resolve_research_primary_train,
)
from mascotrl.reporting.training_telemetry import (
    alias_grad_norm,
    mean_reward_decomp,
    reward_decomp_from_step_info,
)

PROJECTION_MODES = ("soft", "monitor", "off", "hard")

_obs_nan_events = 0


def _assert_obs_finite(
    obs: Any,
    *,
    cfg: Mapping[str, Any] | None = None,
    where: str = "policy_obs",
) -> np.ndarray:
    """Fail closed on NaN/inf in observations before trading on them."""
    global _obs_nan_events
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(arr)):
        _obs_nan_events += 1
        if cfg is not None and isinstance(cfg, dict):
            cfg["obs_nan_events"] = int(cfg.get("obs_nan_events", 0) or 0) + 1
        bad = int(np.sum(~np.isfinite(arr)))
        raise ValueError(
            f"{where} contains {bad} non-finite value(s) (NaN/inf); "
            "refusing to act on corrupted observations"
        )
    return arr


# C5: train_world axis values with a Layer-1 C++ generator (excludes
# "historical" and the meta-value "hybrid_pretrain_finetune", which composes
# one of these with a historical finetune phase -- see synthetic_train_panel).
SYNTHETIC_TRAIN_WORLDS = ("rbergomi", "gbm", "heston", "garch", "sabr")


def synthetic_train_panel(
    cfg: Mapping[str, Any],
    *,
    k: int,
    n_rows: int,
    seed: int,
    world: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """C5: turn a Layer-1 synthetic world into a research-shaped equity panel.

    Runs the same C++ multi-world engine the option-hedge CMDP spine uses
    (:func:`src.simulator.get_world_bundle`) and converts its ``spot_paths``
    (``(n_paths, K, T)``) into daily percentage returns, concatenating paths
    end-to-end until at least ``n_rows`` rows exist (each path is already an
    i.i.d. draw from the world, so concatenation does not smuggle in
    look-ahead). ``factors`` is a synthetic 4-column stand-in (cross-
    sectional mean return plus three zero columns) since the real FF4/PS
    factors have no synthetic-world analogue; residualization against an
    all-zero factor beyond the market column degrades gracefully to a
    market-only regression rather than crashing on shape mismatch.
    """
    from mascotrl.simulator import get_world_bundle, make_identity_cholesky

    w = str(world or cfg.get("train_world") or "rbergomi").lower()
    if w not in SYNTHETIC_TRAIN_WORLDS:
        raise ValueError(
            f"synthetic_train_panel: world={w!r} must be one of {SYNTHETIC_TRAIN_WORLDS}"
        )
    k = int(k)
    n_rows = int(n_rows)
    if k <= 0 or n_rows <= 0:
        raise ValueError(f"synthetic_train_panel requires k>0 and n_rows>0, got k={k} n_rows={n_rows}")

    steps_per_path = max(8, min(int(cfg.get("n_steps", 64) or 64), n_rows + 1))
    n_paths = max(1, -(-n_rows // (steps_per_path - 1)))  # ceil division
    world_cfg = dict(cfg)
    world_cfg.update(
        {
            "train_world": w,
            "n_assets": k,
            "n_steps": steps_per_path,
            "n_paths": n_paths,
            "n_strikes": int(cfg.get("n_strikes", 5) or 5),
            "n_maturities": int(cfg.get("n_maturities", 2) or 2),
            "hurst_exponent": float(cfg.get("hurst_exponent", 0.1) or 0.1),
            "seed": int(seed),
            "force_world_bundle": True,
        }
    )
    bundle = get_world_bundle(world_cfg, cholesky_matrix=make_identity_cholesky(k))
    spots = bundle.get("spot_paths")
    if spots is None:
        raise ValueError(f"train_world={w!r} produced no spot_paths (engine bundle malformed)")
    spots_np = spots.detach().cpu().numpy()
    chunks = []
    for p in range(spots_np.shape[0]):
        path = spots_np[p]  # (K, T)
        r = np.diff(path, axis=1) / np.clip(np.abs(path[:, :-1]), 1e-8, None)
        chunks.append(r.T)  # (T-1, K)
    rets = np.concatenate(chunks, axis=0)
    if rets.shape[0] < n_rows:
        raise ValueError(
            f"synthetic panel rows={rets.shape[0]} < requested n_rows={n_rows} "
            "(increase n_steps/n_paths)"
        )
    rets = np.ascontiguousarray(rets[:n_rows], dtype=np.float64)
    mkt = rets.mean(axis=1, keepdims=True)
    factors = np.ascontiguousarray(
        np.concatenate([mkt, np.zeros((n_rows, 3), dtype=np.float64)], axis=1)
    )
    return rets, factors


def _soft_project(w: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
    """Soft / monitor / off projection: passthrough (credit assignment unlocked).

    Turnover is still recorded by ``HistoricalArmEnv`` regardless of mode;
    ``monitor`` differs from ``hard`` only in that it does not clip.
    """
    del args, kwargs
    return np.asarray(w, dtype=np.float64).reshape(-1)


def _turnover_cap_project(
    w: np.ndarray,
    *,
    t: int | None = None,
    w_prev: np.ndarray | None = None,
    tau: float,
    counter: dict[str, int] | None = None,
) -> np.ndarray:
    """A9: minimum-norm correction onto ``{w : ||w - w_prev||_1 <= tau}``.

    Uniformly shrinks the proposed trade ``w - w_prev`` toward ``w_prev``
    along its own direction until the L1 turnover budget is respected. This
    is the same projection ``src.eval.benchmark_panel._clip_turnover`` uses
    for peer strategies, so the policy and its peers face an identical
    turnover constraint under ``projection_mode: hard``.
    """
    del t
    from mascotrl.policy.cmdp_projector import turnover_cap_project

    return turnover_cap_project(w, w_prev=w_prev, tau=tau, counter=counter)

def build_research_hist_env(
    returns: np.ndarray,
    factors: np.ndarray,
    cfg: Mapping[str, Any],
    *,
    residualizer: Any | None = None,
    dates: Any | None = None,
    rebalance_mask: np.ndarray | None = None,
) -> HistoricalArmEnv:
    """Build HistoricalArmEnv with matched research FrictionSpec."""
    # C1: fail closed on any unregistered spectrum axis value (train_world,
    # architecture, objective, algo) before building the env/agent.
    from mascotrl.spectrum.registry import validate_cfg as _validate_spectrum_cfg

    _validate_spectrum_cfg(cfg)
    resolve_research_primary_train(cfg)
    train_fric, oos_fric = research_train_friction_pair(cfg)
    assert_friction_parity(train_fric, oos_fric)
    # B-COST / L3: cost_in_decision is a parity assertion, not an on/off switch.
    if bool(cfg.get("cost_in_decision", False)):
        eq_bps = float(train_fric.equity_bps)
        hedge_bps = float(train_fric.hedge_leg_bps)
        impact = float(train_fric.execution_impact_coef)
        if eq_bps <= 0.0 and hedge_bps <= 0.0 and impact <= 0.0:
            raise ValueError(
                "cost_in_decision_requires_nonzero_friction: cost_in_decision=true "
                "but FrictionSpec has zero equity_bps, hedge_leg_bps, and impact"
            )
    rets = np.asarray(returns, dtype=np.float64)
    fac = np.asarray(factors, dtype=np.float64)
    k = int(rets.shape[1])
    cfg_local = track_copy(cfg)
    # Phase B: monthly/weekly cadence via rebalance_mask.
    cadence = str(cfg_local.get("rebalance_cadence") or "daily").lower()
    mask = rebalance_mask
    if mask is None and cfg_local.get("_rebalance_mask") is not None:
        mask = np.asarray(cfg_local["_rebalance_mask"], dtype=bool)
    if mask is None and dates is not None:
        from mascotrl.eval.cadence import build_rebalance_mask

        mask = build_rebalance_mask(dates, cadence)
    if cadence not in ("", "daily") and mask is None and dates is None:
        raise ValueError(
            f"rebalance_cadence={cadence!r} requires `_rebalance_mask` or `dates`; "
            "refusing to train with a daily schedule while claiming non-daily cadence"
        )
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.size != rets.shape[0]:
            raise ValueError(
                f"rebalance_mask length {mask.size} != returns T={rets.shape[0]}"
            )
    # W4.2: dynamic (slot-masked) universe arms stamp a (T, K) validity mask
    # onto cfg so CPCV fold slicing (_slice_feature_extras) can carry it
    # alongside _rebalance_mask without threading a new kwarg through every
    # caller of build_research_hist_env.
    slot_valid_mask = cfg_local.get("_slot_valid_mask")
    if slot_valid_mask is not None:
        slot_valid_mask = np.asarray(slot_valid_mask, dtype=bool)
        if slot_valid_mask.shape != (rets.shape[0], k):
            raise ValueError(
                f"_slot_valid_mask shape {slot_valid_mask.shape} != "
                f"(T,K)=({rets.shape[0]},{k})"
            )
    cfg_local["n_assets"] = k
    # Align YAML arm slot counts to the realized panel width (coverage can drop names).
    arm_block = dict(cfg_local.get("arm") or {})
    if arm_block:
        aid = str(arm_block.get("id") or "eq")
        if aid == "eq":
            arm_block["option_slots"] = 0
            arm_block["equity_slots"] = k
        elif aid == "opt":
            arm_block["option_slots"] = k
            arm_block["equity_slots"] = 0
        elif aid == "mix":
            # Keep the configured opt/eq split; only realign if K matches.
            n_opt = int(arm_block.get("option_slots") or (k // 2))
            n_eq = int(arm_block.get("equity_slots") or (k - n_opt))
            if n_opt + n_eq != k:
                n_opt = k // 2
                n_eq = k - n_opt
            arm_block["option_slots"] = n_opt
            arm_block["equity_slots"] = n_eq
        cfg_local["arm"] = arm_block
    # portfolio_arm shorthand: ensure mix slot counts match panel width.
    pa = str(cfg_local.get("portfolio_arm") or "").lower()
    if pa == "mix" and not arm_block:
        cfg_local["n_assets"] = k
    # A10: do not swallow arm_spec_from_cfg errors. It already returns
    # default_arm_spec(n_assets) for an absent ``arm:`` block; anything it
    # raises (e.g. fail_on_load, enabled=false under residual protocol) is
    # an intentional fail-closed gate and must propagate.
    arm = arm_spec_from_cfg(cfg_local)
    if int(arm.n_slots) != k:
        raise ValueError(f"arm.n_slots={arm.n_slots} != returns K={k} after align")
    if residualizer is None:
        y = np.nanmean(rets, axis=1)
        residualizer = freeze_residualizer(
            fit_ff4_residualizer(y, fac, fold_id="research"), "research"
        )
    mode = str(cfg.get("projection_mode") or "soft")
    if mode not in PROJECTION_MODES:
        raise ValueError(f"unknown projection_mode={mode!r}; expected one of {PROJECTION_MODES}")
    from mascotrl.spectrum.policy_mode import apply_turnover_multiplier, resolve_policy_mode

    policy_mode = resolve_policy_mode(cfg_local)
    if mode == "hard":
        turnover_limit = cfg_local.get("turnover_limit")
        if turnover_limit is None:
            raise ValueError(
                "projection_mode='hard' requires cfg['turnover_limit'] (A9: the "
                "cap must be enforced, not decorative)"
            )
        tau = apply_turnover_multiplier(float(turnover_limit), policy_mode)
        turnover_counter = {"steps": 0, "binding_steps": 0}
        project_fn = functools.partial(
            _turnover_cap_project,
            tau=tau,
            counter=turnover_counter,
        )
    else:
        turnover_counter = None
        project_fn = _soft_project
    feature_builder = None
    if bool(cfg.get("use_equity_feature_cube", False)):
        from mascotrl.features.blocks.obs_builder import PanelObservationBuilder

        extras = dict(cfg.get("feature_extras") or {})
        # Fail closed on misaligned panel extras: a fold that trains without
        # ADV/IV/borrow the config claims to use is a silent PIT/feature lie.
        t_len = int(rets.shape[0])
        for key in (
            "dollar_volume",
            "iv",
            "borrow",
            "fundamentals",
            "iv_surface",
            "kelly_images",
            "macro",
            "ohlc",
            "microstructure",
            "fundamentals_pit",
            "sentiment",
            "option_flow",
            "jkp",
        ):
            arr = extras.get(key)
            if arr is None:
                continue
            if isinstance(arr, dict):
                for sk, sv in arr.items():
                    a = np.asarray(sv)
                    if a.ndim >= 1 and int(a.shape[0]) != t_len:
                        raise ValueError(
                            f"feature_extras[{key!r}][{sk!r}] T={a.shape[0]} "
                            f"mismatch panel T={t_len}"
                        )
                    if a.ndim >= 2 and int(a.shape[1]) != k:
                        raise ValueError(
                            f"feature_extras[{key!r}][{sk!r}] K={a.shape[1]} "
                            f"mismatch panel K={k}"
                        )
                continue
            a = np.asarray(arr)
            if a.ndim >= 1 and int(a.shape[0]) != t_len:
                raise ValueError(
                    f"feature_extras[{key!r}] T={a.shape[0]} mismatch panel T={t_len}"
                )
            if key != "macro" and a.ndim >= 2 and int(a.shape[1]) != k:
                raise ValueError(
                    f"feature_extras[{key!r}] K={a.shape[1]} mismatch panel K={k}"
                )
            if key == "macro" and a.ndim == 3 and int(a.shape[1]) != k:
                raise ValueError(
                    f"feature_extras[{key!r}] K={a.shape[1]} mismatch panel K={k}"
                )
        # Stamp YAML excludes into extras for assemble.
        if "feature_groups_exclude" in cfg_local:
            extras["feature_groups_exclude"] = list(
                cfg_local.get("feature_groups_exclude") or []
            )
        if "feature_channels_exclude" in cfg_local:
            extras["feature_channels_exclude"] = list(
                cfg_local.get("feature_channels_exclude") or []
            )
        # Wave 3: if fioracle is enabled but campaign did not pre-attach macro,
        # load here so research train still sees the cube block.
        from mascotrl.data.macro_loader import (
            attach_fioracle_macro_cube,
            fioracle_cfg_from_feature_extras,
        )

        fio_on, _, _ = fioracle_cfg_from_feature_extras(cfg_local)
        if fio_on and extras.get("macro") is None and dates is not None:
            lake = cfg_local.get("lake_root") or cfg_local.get("_lake_root")
            if lake is not None:
                d0 = pd.Timestamp(np.asarray(dates).reshape(-1)[0]).strftime("%Y-%m-%d")
                d1 = pd.Timestamp(np.asarray(dates).reshape(-1)[-1]).strftime("%Y-%m-%d")
                attach_fioracle_macro_cube(
                    cfg_local,
                    lake_base_dir=lake,
                    start_date=d0,
                    end_date=d1,
                    dates=pd.to_datetime(np.asarray(dates).reshape(-1)),
                    out_dir=None,
                    prefer_arctic=False,
                )
                extras = dict(cfg_local.get("feature_extras") or {})
        if bool(cfg.get("include_residual_momentum", False)):
            extras["include_residual_momentum"] = True
            extras["factors"] = fac
        if bool(cfg.get("use_surface_image_encoder", False)):
            # B4: SurfaceImageEncoder needs the raw (11, 34) Kelly IV-surface
            # grid in the cube; fail closed if a fold-realigned kelly_images
            # got dropped above rather than silently training without it.
            if extras.get("kelly_images") is None:
                raise ValueError(
                    "use_surface_image_encoder=true but no aligned "
                    "extras['kelly_images'] survived the T/K fold check; "
                    "pass a (T,K,11,34) array covering this fold's dates."
                )
            extras["include_surface_image_encoder"] = True
        if slot_valid_mask is not None:
            extras["slot_valid_mask"] = slot_valid_mask
        feature_builder = PanelObservationBuilder(
            rets,
            factors=fac if extras.get("include_residual_momentum") else None,
            extras=extras,
            seq_len=int(cfg.get("feature_seq_len", 1) or 1),
            normalize=True,
        )
    env = HistoricalArmEnv(
        returns=rets,
        factors=fac,
        arm=arm,
        friction=train_fric,
        residualizer=residualizer,
        project_fn=project_fn,
        feature_builder=feature_builder,
        rebalance_mask=mask,
        slot_valid_mask=slot_valid_mask,
        reward_mode=_resolve_reward_mode(cfg_local),
        marks=cfg_local.get("_om_marks"),
    )
    env.turnover_cap_counter = turnover_counter
    env.policy_mode = policy_mode
    return env


def _resolve_reward_mode(cfg: Mapping[str, Any]) -> str:
    """D.3: mtm_pnl returns gross MTM minus costs when explicitly selected."""
    if bool(cfg.get("mtm_pnl_reward", False)):
        return "mtm_pnl"
    try:
        from mascotrl.policy.objective_factory import resolve_objective_mode

        if resolve_objective_mode(dict(cfg), default="none") == "mtm_pnl":
            return "mtm_pnl"
    except ImportError:
        pass
    return "residual"


def train_objective_equals_claim_metric(cfg: Mapping[str, Any]) -> bool:
    """True only when training reward aligns with headline total_net Sharpe.

    Default env reward is residual / shaped residual; headline Sharpe is
    ``total_net``. Only ``mtm_pnl`` training approximately matches the claim.
    """
    return _resolve_reward_mode(cfg) == "mtm_pnl"


def _is_ppo_style(agent: Any) -> bool:
    """True when train_epoch accepts old_logprobs / n_epochs / sample_weight / SCR."""
    name = str(getattr(agent, "name", "") or "").lower()
    return name in {"ppo", "cppo", "cppo_omnisafe", "ppo_recurrent"}


def _cfg_num(cfg: Mapping[str, Any], key: str, default: float) -> float:
    """Read a numeric cfg key; honor explicit zeros (do not use ``or default``)."""
    if key not in cfg or cfg[key] is None:
        return float(default)
    return float(cfg[key])


def _agent_policy_module(agent: Any) -> torch.nn.Module | None:
    """Best-effort nn.Module lookup across the single-agent adapters.

    Prefer ``actor`` over ``q`` so DDPG (which has both) checkpoints the
    policy, not the critic. PPO/MCPG expose ``net``; DQN ``q``; SAC/TD3/RRL
    ``actor``. Returns ``None`` for anything else so checkpointing degrades
    to a no-op.
    """
    for attr in ("actor", "net", "q"):
        mod = getattr(agent, attr, None)
        if isinstance(mod, torch.nn.Module):
            return mod
    return None


def _checkpoint_payload(
    agent: Any, cfg: Mapping[str, Any], *, seed: int, episode: int, optimizer_steps: int
) -> dict[str, Any] | None:
    agent_state = None
    if hasattr(agent, "checkpoint_state"):
        agent_state = agent.checkpoint_state()
    net = _agent_policy_module(agent)
    if net is None and agent_state is None:
        return None
    opt = getattr(agent, "opt", None)
    image_digest = str(
        cfg.get("image_digest")
        or cfg.get("_image_digest")
        or os.environ.get("MASCOTRL_CONTAINER_DIGEST")
        or ""
    ).strip()
    return {
        "policy": net.state_dict() if net is not None else None,
        "agent_state": agent_state,
        "optimizer": opt.state_dict() if opt is not None else None,
        "seed": int(seed),
        "fold_id": cfg.get("_fold_id"),
        "run_config_hash": cfg.get("_run_config_hash"),
        "image_digest": image_digest or None,
        "episode": int(episode),
        "optimizer_steps": int(optimizer_steps),
    }


def _save_checkpoint(
    agent: Any, cfg: Mapping[str, Any], *, seed: int, episode: int, optimizer_steps: int
) -> None:
    """W3.2: intra-fold checkpoint so a crashed multi-episode fold can be
    resumed from the last saved episode instead of restarting the fold.
    """
    ckpt_dir = cfg.get("_checkpoint_dir")
    if not ckpt_dir:
        return
    payload = _checkpoint_payload(
        agent, cfg, seed=seed, episode=episode, optimizer_steps=optimizer_steps
    )
    if payload is None:
        if isinstance(cfg, dict):
            cfg["checkpoint_payload_empty"] = True
        ckpt_every = cfg.get("checkpoint_every_n_episodes")
        if ckpt_every is not None and int(ckpt_every) > 0:
            raise RuntimeError(
                "checkpoint_payload_empty: agent exposes no checkpoint_state or "
                "policy module but checkpoint_every_n_episodes was explicitly set"
            )
        return
    d = Path(str(ckpt_dir))
    d.mkdir(parents=True, exist_ok=True)
    fold_tag = payload["fold_id"] if payload["fold_id"] is not None else "na"
    torch.save(payload, d / f"fold{fold_tag}_seed{int(seed)}_ep{int(episode):05d}.pt")


def _discover_latest_checkpoint(
    ckpt_dir: Path | str,
    seed: int,
    fold_id: int,
    run_config_hash: str | None,
) -> Path | None:
    """Return the latest checkpoint matching one seed/fold/config cell."""
    directory = Path(ckpt_dir)
    if not directory.is_dir():
        return None
    latest: tuple[int, Path] | None = None
    for path in directory.glob(f"fold{int(fold_id)}_seed{int(seed)}_ep*.pt"):
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if int(blob.get("seed", -1)) != int(seed):
            continue
        if int(blob.get("fold_id", -1)) != int(fold_id):
            continue
        if blob.get("run_config_hash") != run_config_hash:
            continue
        episode = int(blob.get("episode", -1))
        if latest is None or episode > latest[0]:
            latest = (episode, path)
    return latest[1] if latest is not None else None


def prune_fold_checkpoints(
    ckpt_dir: Path | str,
    *,
    keep_latest: int = 1,
    fold_id: int | None = None,
    seed: int | None = None,
) -> int:
    """Delete older fold checkpoints, keeping the newest ``keep_latest`` by mtime.

    When ``fold_id`` and ``seed`` are set, only files matching
    ``fold{fold_id}_seed{seed}_ep*.pt`` are considered so a shared checkpoint
    directory cannot wipe sibling folds.
    """
    directory = Path(ckpt_dir)
    if not directory.is_dir():
        return 0
    if fold_id is not None and seed is not None:
        pattern = f"fold{int(fold_id)}_seed{int(seed)}_ep*.pt"
    else:
        pattern = "*.pt"
    files = sorted(
        directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True
    )
    keep = max(0, int(keep_latest))
    deleted = 0
    for path in files[keep:]:
        try:
            path.unlink()
            deleted += 1
        except OSError:
            continue
    return deleted


def _maybe_resume_checkpoint(agent: Any, cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    """W3.2: load a prior intra-fold checkpoint, fail closed on config drift.

    Mirrors ``scripts/train_happo.py``'s full-run ``--resume`` path (load
    state dict, refuse silently loading weights trained under a different
    run configuration) but scoped to one CPCV fold's checkpoint instead of
    a whole training run.
    """
    resume_path = cfg.get("_resume_checkpoint")
    if not resume_path:
        return None
    p = Path(str(resume_path))
    if not p.exists():
        return None
    blob = torch.load(p, map_location="cpu", weights_only=False)
    expected_hash = cfg.get("_run_config_hash")
    if expected_hash is None:
        raise RuntimeError(
            "checkpoint resume requires cfg['_run_config_hash']; "
            "refusing to load without a config fingerprint"
        )
    if blob.get("run_config_hash") != expected_hash:
        raise RuntimeError(
            "checkpoint run_config_hash mismatch (checkpoint="
            f"{blob.get('run_config_hash')!r} != cfg={expected_hash!r}); "
            "refusing to resume weights trained under a different config"
        )
    expected_digest = str(
        cfg.get("image_digest")
        or cfg.get("_image_digest")
        or os.environ.get("MASCOTRL_CONTAINER_DIGEST")
        or ""
    ).strip()
    stored_digest = str(blob.get("image_digest") or "").strip()
    if stored_digest and expected_digest and stored_digest != expected_digest:
        raise RuntimeError(
            f"digest_mismatch: checkpoint image_digest={stored_digest!r} "
            f"!= running={expected_digest!r}"
        )
    # Prefer full agent checkpoint_state when present (SAC/TD3/DDPG/RRL).
    if hasattr(agent, "load_checkpoint_state") and blob.get("agent_state") is not None:
        agent.load_checkpoint_state(blob["agent_state"])
    else:
        net = _agent_policy_module(agent)
        if net is not None and blob.get("policy") is not None:
            net.load_state_dict(blob["policy"])
        opt = getattr(agent, "opt", None)
        if opt is not None and blob.get("optimizer") is not None:
            opt.load_state_dict(blob["optimizer"])
    return blob


def train_research_hist(
    returns: np.ndarray,
    factors: np.ndarray,
    cfg: Mapping[str, Any],
    *,
    seed: int = 0,
    agent: Any | None = None,
) -> dict[str, Any]:
    """Roll panel(s) through HistoricalArmEnv; correct PPO on collected traj.

    ``agent`` may be passed to warm-start across folds (Phase D).
    """
    import os

    n = int(
        os.environ.get("TORCH_NUM_THREADS")
        or os.environ.get("MASCOTRL_THREADS_PER_WORKER")
        or "0"
    )
    if n > 0:
        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(max(1, min(n, 2)))
        except RuntimeError:
            pass  # already set
    primary = resolve_research_primary_train(cfg)
    if primary not in RESEARCH_PRIMARY_ALLOWED:
        raise ValueError(f"expected one of {sorted(RESEARCH_PRIMARY_ALLOWED)}, got {primary}")
    if primary == "hybrid_pretrain_finetune":
        primary = RESEARCH_PRIMARY_HIST
    policy = str(cfg.get("policy") or "single_agent")
    if policy not in ("single_agent", "ppo"):
        raise ValueError(f"unknown policy={policy!r}; allowed=['single_agent', 'ppo']")
    # C1: fail closed on any spectrum key with an unregistered value before
    # spending a single training step on it.
    axes = validate_cfg(cfg)
    architecture = axes["architecture"]
    algo = axes["algo"]
    env = build_research_hist_env(returns, factors, cfg)
    k = int(env.K)
    obs0, _ = env.reset(seed=seed)
    obs_dim = int(np.asarray(obs0, dtype=np.float32).reshape(-1).size)
    # W2.2: seed the global RNGs before agent construction, not after, so
    # two different seeds actually produce two different sets of initial
    # network weights (orthogonal_init draws from the global torch RNG).
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    from mascotrl.policy.sb3_adapter import resolve_rl_backend

    _rl_backend = resolve_rl_backend(cfg)
    agent_was_provided = agent is not None
    if agent is None:
        # C2 / D.2: architecture axis via shared build_policy_body for
        # ppo/sac/td3/ddpg. mlp keeps the flat body; gru/lstm/transformer/
        # mamba need the asset-major feature cube.
        hidden = int(cfg.get("ppo_hidden", 64) or 64)
        lr = float(cfg.get("lr", 3e-4))
        gamma = float(cfg.get("gamma", 0.99) or 0.99)
        arch_kwargs: dict[str, Any] = {"architecture": architecture}
        if architecture != "mlp":
            fb = getattr(env, "feature_builder", None)
            if fb is None:
                raise ValueError(
                    f"architecture={architecture!r} requires "
                    "use_equity_feature_cube=true (the asset-major "
                    "(K, seq, C) layout is only defined by PanelObservationBuilder)"
                )
            arch_kwargs.update(
                {
                    "num_assets": k,
                    "d_model": int(fb.obs_channels_per_asset),
                    "seq_len": int(fb.seq_len),
                    "d_state": int(cfg.get("d_state", 16) or 16),
                    "share_temporal_encoder": bool(
                        cfg.get("share_temporal_encoder", True)
                    ),
                }
            )
            if architecture in ("mamba", "mamba2") and cfg.get("mamba_chunk_size") is not None:
                arch_kwargs["chunk_size"] = int(cfg.get("mamba_chunk_size"))
            if bool(cfg.get("use_surface_image_encoder", False)):
                arch_kwargs["use_surface_image_encoder"] = True
                arch_kwargs["image_channels"] = 11 * 34
                arch_kwargs["surface_image_embed_dim"] = int(
                    cfg.get("surface_image_embed_dim", 16) or 16
                )
        # C4: algo axis reachable in research training, not only resolved
        # and discarded. Each adapter keeps its own constructor kwargs
        # (single-agent adapters are not all parameterized alike -- e.g.
        # DQN has no gae_lambda, RRL has no critic/value_coef).
        if algo in ("ppo", "cppo", "cppo_omnisafe"):
            if algo == "cppo_omnisafe":
                _agent_id = "cppo_omnisafe"
            elif algo == "cppo":
                _agent_id = "cppo"
            else:
                _agent_id = "ppo"
            _ppo_kw: dict[str, Any] = dict(
                entropy_coef=float(cfg.get("entropy_coef", 0.02) or 0.0),
                gamma=gamma,
                gae_lambda=float(cfg.get("gae_lambda", 0.95) or 0.95),
                weight_head=str(cfg.get("weight_head") or "softmax"),
                hidden=hidden,
                clip_eps=float(cfg.get("clip_eps", 0.2) or 0.2),
                actor_final_gain=float(cfg.get("actor_final_gain", 0.1) or 0.1),
                weight_head_temperature=float(
                    cfg.get("weight_head_temperature", 1.0) or 1.0
                ),
                weight_head_tilt_gain=float(
                    cfg.get("weight_head_tilt_gain", 1.0) or 1.0
                ),
                **arch_kwargs,
            )
            if _agent_id in ("cppo", "cppo_omnisafe"):
                _ppo_kw.update(
                    cvar_alpha=float(cfg.get("cvar_alpha", 0.95) or 0.95),
                    cvar_k_ratio=float(cfg.get("cvar_k_ratio", 0.2) or 0.2),
                    nu_lr=float(cfg.get("nu_lr", 0.01) or 0.01),
                    nu_delay=float(cfg.get("nu_delay", 0.2) or 0.2),
                )
                if _agent_id == "cppo_omnisafe":
                    _ppo_kw["omnisafe_algo"] = str(
                        cfg.get("omnisafe_algo") or "cppo_pid"
                    )
            agent = make_single_agent(
                _agent_id,
                obs_dim=obs_dim,
                action_dim=k,
                lr=lr,
                rl_backend=_rl_backend if _agent_id == "ppo" else "custom",
                **_ppo_kw,
            )
        elif algo in ("sac", "td3"):
            agent = make_single_agent(
                algo,
                obs_dim=obs_dim,
                action_dim=k,
                lr=lr,
                rl_backend=_rl_backend,
                gamma=gamma,
                hidden=hidden,
                weight_head=str(cfg.get("weight_head") or "softmax"),
                weight_head_tilt_gain=float(
                    cfg.get("weight_head_tilt_gain", 1.0) or 1.0
                ),
                **arch_kwargs,
            )
        elif algo == "ddpg":
            agent = make_single_agent(
                algo,
                obs_dim=obs_dim,
                action_dim=k,
                lr=lr,
                rl_backend=_rl_backend,
                gamma=gamma,
                hidden=hidden,
                weight_head=str(cfg.get("weight_head") or "tanh_l1"),
                weight_head_tilt_gain=float(
                    cfg.get("weight_head_tilt_gain", 1.0) or 1.0
                ),
                **arch_kwargs,
            )
        elif algo == "mcpg":
            agent = make_single_agent(
                "mcpg",
                obs_dim=obs_dim,
                action_dim=k,
                lr=lr,
                rl_backend="custom",
                gamma=gamma,
                hidden=hidden,
                entropy_coef=float(cfg.get("entropy_coef", 0.01) or 0.0),
                weight_head=str(cfg.get("weight_head") or "tanh_l1"),
                actor_final_gain=float(cfg.get("actor_final_gain", 0.1) or 0.1),
                weight_head_temperature=float(
                    cfg.get("weight_head_temperature", 1.0) or 1.0
                ),
                **arch_kwargs,
            )
        elif algo == "rrl":
            agent = make_single_agent(
                "rrl",
                obs_dim=obs_dim,
                action_dim=k,
                lr=lr,
                rl_backend="custom",
                hidden=hidden,
                eta=float(cfg.get("diff_sharpe_eta", 0.01) or 0.01),
                weight_head=str(cfg.get("weight_head") or "tanh_l1"),
                **arch_kwargs,
            )
        elif algo == "dqn":
            # Wave 5: registry marks requires_discrete; refuse non-mlp here too.
            if architecture != "mlp":
                raise ValueError(
                    f"algo='dqn' requires_discrete; architecture={architecture!r} "
                    "is not wired (discrete Q heads remain flat-MLP only)"
                )
            agent = make_single_agent(
                "dqn",
                obs_dim=obs_dim,
                action_dim=k,
                lr=lr,
                rl_backend="custom",
                gamma=gamma,
                hidden=hidden,
            )
        elif algo == "happo":
            raise ValueError(
                "algo='happo' is multi-agent; spectrum campaign routes it "
                "before train_research_hist (see run_spectrum_campaign)"
            )
        else:  # pragma: no cover - validate_cfg already restricts algo's range
            raise ValueError(f"unhandled algo={algo!r}")

    # W3.2: intra-fold checkpointing. Resume only when the caller did not
    # pass a warm-started agent (hybrid pretrain+finetune must keep pretrain
    # weights). Episode skip uses the resumed blob's episode counter.
    resume_blob: dict[str, Any] | None = None
    if cfg.get("_checkpoint_dir"):
        Path(str(cfg["_checkpoint_dir"])).mkdir(parents=True, exist_ok=True)
    if cfg.get("_resume_checkpoint") and not agent_was_provided:
        resume_blob = _maybe_resume_checkpoint(agent, cfg)
    start_ep = int(resume_blob.get("episode", 0)) if resume_blob else 0

    target_steps = int(cfg.get("train_env_steps", 0) or 0)
    n_episodes = int(cfg.get("train_episodes", 1) or 1)
    if target_steps > 0:
        ep_len = max(1, int(env.T) - 2)
        n_episodes = max(n_episodes, int(np.ceil(target_steps / ep_len)))
    # W2.3: train_updates_per_fold repeats the collect-then-train_epoch
    # cycle on a fresh trajectory from the (now partially trained) agent,
    # instead of a single trajectory/update per fold. Default 1 reproduces
    # the pre-existing single-update behaviour exactly.
    train_updates_per_fold = max(1, int(cfg.get("train_updates_per_fold", 1) or 1))

    curve: list[dict[str, float]] = []

    # C3: objective axis reachable from the research PPO path, not only
    # HAPPO. `reward: differential_sharpe` (legacy short form) and
    # `objective: differential_sharpe`/`mikkila_asym` both select the
    # dense-reward transform applied inline to each env step; the six
    # episode_weight objectives (mean_std_cao, meanvar_kolm, cvar_ru,
    # entropic_oce, smse, rsqp) reweight the PPO advantage per finished
    # episode (see sample_weight below), the research-PPO analogue of
    # HAPPO's score-function episode weights.
    objective_mode = resolve_objective_mode(cfg, default="none")
    from mascotrl.eval.yaml_honesty import refuse_rrl_double_dsr

    refuse_rrl_double_dsr({**dict(cfg), "algo": algo, "objective": objective_mode})
    # Episode-weight spectrum objectives are primary by default so the OFAT
    # axis actually reaches the PPO update (cherrypick YAMLs emit the flag;
    # absent key still activates for known episode-weight modes).
    if "objective_primary" in cfg:
        objective_primary = bool(cfg.get("objective_primary"))
    else:
        from mascotrl.policy.objective_factory import _EPISODE_WEIGHT_MODES

        objective_primary = objective_mode in _EPISODE_WEIGHT_MODES
    obj_path = objective_gradient_path_for(objective_mode, objective_primary)
    use_ds = str(cfg.get("reward") or "") == "differential_sharpe" or (
        obj_path == "dense_reward" and objective_mode == "differential_sharpe"
    )
    if use_ds and obj_path == "episode_weight":
        raise ValueError(
            "reward/objective differential_sharpe cannot stack with "
            "episode_weight objectives (double shaping)"
        )
    use_mikkila = obj_path == "dense_reward" and objective_mode == "mikkila_asym"
    use_sdr = obj_path == "dense_reward" and objective_mode == "sdr_composite"
    mikkila_xi = _cfg_num(cfg, "mikkila_xi", 1.0)
    n_minibatches = int(_cfg_num(cfg, "n_minibatches", 4.0))
    if n_minibatches < 1:
        n_minibatches = 1
    ckpt_every_n = int(cfg.get("checkpoint_every_n_episodes", 0) or 0)
    reward_weights = dict(cfg.get("reward_weights") or {})
    risk_cfg = dict(cfg.get("risk") or {})
    epochs = int(cfg.get("train_epochs", 1) or 1)
    # RC2: policy_mode risk-aversion must reach episode_weights (turnover
    # scaling alone left carry/crisis/inflation half-implemented).
    from mascotrl.spectrum.policy_mode import (
        apply_risk_aversion,
        resolve_policy_mode,
        resolve_term_spread_z_for_train,
    )

    _pm = resolve_policy_mode(cfg)
    _term_spread_z = resolve_term_spread_z_for_train(cfg)
    _scaled_cao_c = apply_risk_aversion(
        float(risk_cfg.get("cao_c", cfg.get("cao_c", 1.5))),
        _pm,
        term_spread_z=_term_spread_z,
    )
    _scaled_kappa = apply_risk_aversion(
        float(risk_cfg.get("kappa", cfg.get("kappa", 1.0))),
        _pm,
        term_spread_z=_term_spread_z,
    )

    total_opt_steps = 0
    last_stats: dict[str, float] = {}
    rew_all: list[float] = []
    any_trained = False
    realized_costs: list[float] = []
    realized_option_spreads: list[float] = []

    for update_idx in range(train_updates_per_fold):
        obs_list: list[np.ndarray] = []
        act_list: list[np.ndarray] = []
        logp_list: list[float] = []
        rew_list: list[float] = []
        next_list: list[np.ndarray] = []
        terminated_list: list[float] = []
        rebal_list: list[float] = []
        visited_t_list: list[int] = []
        ep_lengths: list[int] = []
        ep_totals: list[float] = []

        for ep in range(max(1, n_episodes)):
            global_ep = update_idx * max(1, n_episodes) + ep
            if global_ep < start_ep:
                continue
            ds = DifferentialSharpe(eta=float(cfg.get("diff_sharpe_eta", 0.01))) if use_ds else None
            obs, _ = env.reset(seed=int(seed) + global_ep)
            terminated = False
            truncated = False
            ep_rew: list[float] = []
            ep_decomp: list[dict[str, float]] = []
            while not (terminated or truncated):
                if target_steps > 0 and len(rew_list) >= target_steps:
                    # Budget exhausted: stop collecting, but do not silently drop
                    # the pending transition (A8). The prior code broke here
                    # before stepping/appending, so the last stored transition
                    # was already several steps stale.
                    break
                obs_clean = _assert_obs_finite(obs, cfg=cfg, where="policy_obs")
                obs_t = torch.as_tensor(obs_clean, dtype=torch.float32).unsqueeze(0)
                # RC6: expose EW-on-mask baseline to sparse_tilt / dirichlet heads.
                fb = getattr(env, "feature_builder", None)
                if fb is not None:
                    from mascotrl.features.blocks.obs_builder import equal_weight_on_mask

                    mask = getattr(fb, "_slot_mask", None)
                    if mask is None:
                        w_base_np = np.full(
                            int(env.K), 1.0 / max(int(env.K), 1), dtype=np.float64
                        )
                    else:
                        w_base_np = equal_weight_on_mask(mask)
                    agent._last_w_base = torch.as_tensor(
                        w_base_np, dtype=torch.float32
                    )
                with torch.no_grad():
                    if hasattr(agent, "act_and_logp_raw"):
                        raw, logp = agent.act_and_logp_raw(obs_t, deterministic=False)
                        w_t = agent.raw_to_weights(raw)
                        logp_v = float(logp.reshape(-1)[0].item())
                        raw_np = raw.detach().cpu().numpy().reshape(-1)
                    else:
                        action = agent.act(obs_t, deterministic=False)
                        w_t = action
                        logp_v = 0.0
                        raw_np = action.detach().cpu().numpy().reshape(-1)
                w = np.nan_to_num(w_t.detach().cpu().numpy().reshape(-1), nan=0.0)
                denom = float(np.sum(np.abs(w)))
                if denom > 1e-8:
                    w = w / denom
                # else: keep zero vector (intentional flat / no exposure)
                # RC5: capture rebalance flag at the step index before env.step advances t.
                rebal_today = True
                rm = getattr(env, "rebalance_mask", None)
                t_now = int(getattr(env, "t", 0) or 0)
                if rm is not None and 0 <= t_now < len(rm):
                    rebal_today = bool(rm[t_now])
                visited_t_list.append(t_now)
                next_obs, reward, terminated, truncated, _info = env.step(w)
                realized_costs.append(float(_info.get("cost", 0.0) or 0.0))
                # option_spread is not always in info; cost covers trading drag.
                train_r = float(reward)
                if ds is not None:
                    train_r = float(ds.step(train_r))
                elif use_mikkila:
                    train_r = mikkila_asym_reward(train_r, xi=mikkila_xi)
                elif use_sdr:
                    train_r = sdr_composite_reward(
                        train_r, weights=reward_weights
                    )
                ep_decomp.append(
                    reward_decomp_from_step_info(
                        _info if isinstance(_info, Mapping) else {},
                        train_reward=train_r,
                    )
                )
                obs_list.append(obs_clean)
                act_list.append(np.asarray(raw_np, dtype=np.float32))
                logp_list.append(logp_v)
                rew_list.append(train_r)
                ep_rew.append(train_r)
                rebal_list.append(1.0 if rebal_today else 0.0)
                next_list.append(
                    _assert_obs_finite(next_obs, cfg=cfg, where="next_obs")
                )
                # GAE masks on terminated only; truncations bootstrap (HAPPO WP1).
                # Budget cutoffs are artificial collection limits, not absorbing states.
                terminated_list.append(1.0 if terminated else 0.0)
                obs = next_obs
                if target_steps > 0 and len(rew_list) >= target_steps:
                    truncated = True
            curve_entry: dict[str, float] = {
                "episode": float(global_ep),
                "mean_reward": float(np.mean(ep_rew)) if ep_rew else float("nan"),
                "n_steps": float(len(ep_rew)),
            }
            curve_entry.update(mean_reward_decomp(ep_decomp))
            curve.append(curve_entry)
            ep_lengths.append(len(ep_rew))
            ep_totals.append(float(np.sum(ep_rew)) if ep_rew else 0.0)
            if ckpt_every_n > 0 and (global_ep + 1) % ckpt_every_n == 0:
                _save_checkpoint(
                    agent, cfg, seed=seed, episode=global_ep + 1, optimizer_steps=total_opt_steps
                )

        if obs_list:
            # C4: PPO-family train_epoch accepts importance-sampling kwargs
            # (old_logprobs/n_epochs/n_minibatches) and episode_weight sample_weight.
            # MCPG may also expose act_and_logp_raw so the collect path stores
            # pre-head logits, but its train_epoch stays on the plain signature.
            is_ppo_style = _is_ppo_style(agent)
            batch = dict(
                obs=torch.as_tensor(np.stack(obs_list), dtype=torch.float32),
                actions=torch.as_tensor(np.stack(act_list), dtype=torch.float32),
                rewards=torch.as_tensor(rew_list, dtype=torch.float32),
                next_obs=torch.as_tensor(np.stack(next_list), dtype=torch.float32),
                dones=torch.as_tensor(terminated_list, dtype=torch.float32),
            )
            if is_ppo_style:
                batch["old_logprobs"] = torch.as_tensor(logp_list, dtype=torch.float32)
                batch["n_epochs"] = max(1, epochs)
                batch["n_minibatches"] = max(1, n_minibatches)
                # RC5: zero PPO advantages on non-rebalance days.
                if any(v < 0.5 for v in rebal_list):
                    batch["policy_step_mask"] = torch.as_tensor(
                        rebal_list, dtype=torch.bool
                    )
            if obj_path == "episode_weight":
                if not (ep_totals and sum(ep_lengths) == len(obs_list)):
                    raise RuntimeError(
                        "episode_weight length mismatch: "
                        f"sum(ep_lengths)={sum(ep_lengths)} != len(obs_list)={len(obs_list)}"
                    )
                if not is_ppo_style:
                    raise ValueError(
                        f"objective={objective_mode!r} requires_episode_returns "
                        f"and is only wired for algo='ppo' today; got algo={algo!r}"
                    )
                G = torch.as_tensor(ep_totals, dtype=torch.float32)
                obj_w = episode_weights(
                    objective_mode,
                    G,
                    cao_c=_scaled_cao_c,
                    kappa=_scaled_kappa,
                    alpha=float(risk_cfg.get("alpha", cfg.get("alpha", 0.95))),
                    lam=float(
                        risk_cfg.get("lambda", risk_cfg.get("lam", cfg.get("lam", 1.0)))
                    ),
                )
                parts: list[float] = []
                for w, length in zip(obj_w.tolist(), ep_lengths):
                    parts.extend([float(w)] * length)
                batch["sample_weight"] = torch.as_tensor(parts, dtype=torch.float32)
            if is_ppo_style:
                from mascotrl.eval.scr_critic import resolve_scr_mix

                scr_mode, scr_beta = resolve_scr_mix(cfg)
                batch["scr_mix"] = scr_mode
                batch["scr_beta"] = float(scr_beta)
                last_stats = agent.train_epoch(**batch)
                total_opt_steps += int(last_stats.get("optimizer_steps", 0) or 0)
            else:
                # Off-policy (SAC/TD3/DDPG) may repeat gradient steps on the
                # collected batch. On-policy MCPG/RRL must stay single-shot:
                # re-scoring log pi_new(a_old) without importance sampling
                # biases the REINFORCE gradient.
                agent_name = str(getattr(agent, "name", "") or "").lower()
                if agent_name in {"mcpg", "rrl"}:
                    n_repeat = 1
                else:
                    n_repeat = max(1, epochs) * max(1, n_minibatches)
                for _ in range(n_repeat):
                    last_stats = agent.train_epoch(**batch)
                    total_opt_steps += 1
                if agent_name in {"mcpg", "rrl"}:
                    last_stats["on_policy_single_shot"] = True
            last_stats = alias_grad_norm(last_stats)
            last_stats["objective_gradient_path"] = obj_path
            # RC6: reward-to-noise = max feasible EW tilt edge vs reward std.
            # Align the return panel to visited env timesteps (not full T).
            if rew_list:
                from mascotrl.eval.reward_noise import reward_to_noise_diagnostic

                tau = float(cfg.get("turnover_limit") or 0.05)
                panel_rets = getattr(env, "returns", None)
                if panel_rets is not None and visited_t_list:
                    idxs = np.asarray(visited_t_list, dtype=np.int64)
                    t_max = int(np.asarray(panel_rets).shape[0])
                    idxs = idxs[(idxs >= 0) & (idxs < t_max)]
                    panel_slice = (
                        np.asarray(panel_rets, dtype=np.float64)[idxs]
                        if idxs.size
                        else np.asarray(panel_rets, dtype=np.float64)[:0]
                    )
                    diag = reward_to_noise_diagnostic(
                        panel_slice, rew_list, turnover_limit=tau
                    )
                elif panel_rets is not None:
                    n = len(rew_list)
                    panel_arr = np.asarray(panel_rets, dtype=np.float64)
                    panel_slice = panel_arr[:n] if panel_arr.shape[0] >= n else panel_arr
                    diag = reward_to_noise_diagnostic(
                        panel_slice, rew_list, turnover_limit=tau
                    )
                else:
                    rew_arr = np.asarray(rew_list, dtype=np.float64)
                    diag = {
                        "reward_std": float(np.std(rew_arr)),
                        "reward_concentrated_vs_ew_gap": float("nan"),
                        "reward_signal_to_noise": float("nan"),
                        "reward_unlearnable": False,
                    }
                last_stats.update(diag)
                if diag.get("reward_noise_warning"):
                    import logging

                    logging.getLogger(__name__).warning(
                        "%s", diag["reward_noise_warning"]
                    )
            if curve:
                # Attach episode-mean reward decomposition to train_stats.
                for k, v in curve[-1].items():
                    if str(k).startswith("reward_") or k == "mean_reward":
                        last_stats[k] = float(v)
            log_std = getattr(getattr(agent, "net", None), "log_std", None)
            if log_std is None:
                log_std = getattr(agent, "log_std", None)
            if log_std is not None:
                last_stats["log_std_mean"] = float(
                    torch.as_tensor(log_std).detach().float().mean().cpu()
                )
            if curve and last_stats.get("entropy") is not None:
                curve[-1]["entropy"] = float(last_stats["entropy"])
            any_trained = True
        rew_all.extend(rew_list)

    if any_trained and hasattr(agent, "freeze_obs_norm"):
        agent.freeze_obs_norm()

    min_steps_floor = int(cfg.get("min_optimizer_steps", 0) or 0)
    if min_steps_floor > 0 and total_opt_steps < min_steps_floor:
        raise RuntimeError(
            f"train budget fail-closed: optimizer_steps={total_opt_steps} "
            f"< min_optimizer_steps={min_steps_floor}"
        )

    total_episodes = int(n_episodes) * train_updates_per_fold

    # W3.2: always checkpoint the final trained state, independent of
    # checkpoint_every_n_episodes, so a fold's finished result survives a
    # crash immediately after training (e.g. during eval rollout below).
    _save_checkpoint(
        agent, cfg, seed=seed, episode=total_episodes, optimizer_steps=total_opt_steps
    )

    mean_realized_cost = (
        float(np.mean(realized_costs)) if realized_costs else 0.0
    )
    has_option_slots = int(getattr(env.arm, "option_slots", 0) or 0) > 0
    friction_applied = mean_realized_cost > 0.0
    if has_option_slots and bool(getattr(env.friction, "om_touch_enabled", False)):
        # Option arms must realize nonzero trading drag under OM-touch.
        friction_applied = friction_applied and mean_realized_cost > 0.0
    cost_in_decision_realized = bool(cfg.get("cost_in_decision", False)) and (
        mean_realized_cost > 0.0
    )
    turnover_counter = getattr(env, "turnover_cap_counter", None) or {}
    turnover_steps = int(turnover_counter.get("steps", 0))
    turnover_bindings = int(turnover_counter.get("binding_steps", 0))
    return {
        "primary_train": primary,
        "policy": "single_agent",
        "rl_backend": str(getattr(agent, "backend", None) or _rl_backend),
        "rl_backend_requested": _rl_backend,
        "projection_mode": str(cfg.get("projection_mode") or "soft"),
        "reward": str(cfg.get("reward") or "residual_pnl"),
        "objective": objective_mode,
        "objective_gradient_path": obj_path,
        "estimand_id": str(cfg.get("estimand_id") or ""),
        "train_objective_equals_claim_metric": train_objective_equals_claim_metric(cfg),
        "friction_applied": bool(friction_applied),
        "cost_in_decision_realized": bool(cost_in_decision_realized),
        "mean_realized_cost": float(mean_realized_cost),
        "n_steps": len(rew_all),
        "n_episodes": total_episodes,
        "train_updates_per_fold": train_updates_per_fold,
        "mean_reward": float(np.mean(rew_all)) if rew_all else float("nan"),
        "train_stats": last_stats,
        "optimizer_steps": total_opt_steps,
        "turnover_cap_projection_steps": turnover_steps,
        "turnover_cap_binding_steps": turnover_bindings,
        "turnover_cap_binding_fraction": (
            float(turnover_bindings / turnover_steps) if turnover_steps else 0.0
        ),
        "learning_curve": curve,
        "agent": agent,
        "env": env,
    }
