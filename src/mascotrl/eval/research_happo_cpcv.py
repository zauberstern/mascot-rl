"""HAPPO CPCV runner for narrative / full-budget spectrum cells."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from mascotrl.eval.collapse_guard import collapse_guard
from mascotrl.eval.cpcv import CPCVConfig, CPCVFold, _CPCV_FOLD_AUX_KEY, run_cpcv
from mascotrl.eval.cpcv_backend import resolve_use_purgedcv
from mascotrl.eval.research_alpha_cpcv import (
    _indices_for_windows,
    _reconstruct_path0_aux_series,
)
from mascotrl.policy.cmdp_config import build_step_costs, resolve_cmdp_cfg
from mascotrl.policy.objective_factory import build_risk_objective
from mascotrl.policy.trainer import HAPPOTrainer, TrainBatch
from mascotrl.reporting.capital_gates import PROJECTION_K_CEILING

_COORD_SCALAR_KEYS = (
    "proj_gap",
    "proj_penalty",
    "exec_turnover",
    "exec_weight_l1",
    "action_l1",
    "approx_kl",
    "clip_frac",
    "entropy",
    "teamtr_skips",
    "teamtr_enabled",
    "cmdp_lambda",
    "cmdp_j_c_violation_frac",
)


def happo_progress_line(kind: str, **fields: object) -> str:
    """CloudWatch-visible progress lines for narrative HAPPO (no silent SCS grind)."""
    if kind == "cpcv_train_start":
        return (
            f"phase=cpcv_train start seed={fields['seed']} "
            f"n_splits={fields['n_splits']} n_test_groups={fields['n_test_groups']}"
        )
    if kind == "cpcv_train_done":
        return f"phase=cpcv_train done seed={fields['seed']}"
    if kind == "train_step":
        return (
            f"phase=happo_train seed={fields['seed']} "
            f"ep={fields['ep']}/{fields['n_episodes']} "
            f"step={fields['step']}/{fields['max_steps']}"
        )
    if kind == "train_ep":
        return (
            f"phase=happo_train_ep seed={fields['seed']} "
            f"ep={fields['ep']}/{fields['n_episodes']} "
            f"backend={fields.get('backend', 'unknown')}"
        )
    raise ValueError(f"unknown happo progress kind={kind}")


def _eval_happo_panel_payload(
    *,
    pnl: Mapping[str, float],
    dates: Sequence[str],
    weights: np.ndarray,
    turnovers: np.ndarray | Sequence[float],
    s_delta: np.ndarray | Sequence[float] | None = None,
    s_turn: np.ndarray | Sequence[float] | None = None,
) -> dict[str, Any]:
    """Build CPCV fold return: date→pnl floats plus ``__aux__`` OOS records."""
    w = np.asarray(weights, dtype=np.float64)
    turns = np.asarray(turnovers, dtype=np.float64).reshape(-1)
    sd = None if s_delta is None else np.asarray(s_delta, dtype=np.float64).reshape(-1)
    st = None if s_turn is None else np.asarray(s_turn, dtype=np.float64).reshape(-1)
    aux: dict[str, dict[str, Any]] = {}
    n = min(len(dates), int(w.shape[0]), int(turns.size))
    for i in range(n):
        ds = str(dates[i])
        row: dict[str, Any] = {
            "weights": [float(x) for x in w[i].tolist()],
            "turnover": float(turns[i]),
        }
        if sd is not None and i < sd.size:
            row["s_delta"] = float(sd[i])
        if st is not None and i < st.size:
            row["s_turn"] = float(st[i])
        aux[ds] = row
    out: dict[str, Any] = {str(k): float(v) for k, v in dict(pnl).items()}
    out[_CPCV_FOLD_AUX_KEY] = aux
    return out


def _agent_order_entropy(orders: Sequence[Any]) -> float | None:
    flat: list[int] = []
    for order in orders:
        if isinstance(order, (list, tuple)):
            flat.extend(int(x) for x in order)
    if not flat:
        return None
    _vals, counts = np.unique(np.asarray(flat, dtype=np.int64), return_counts=True)
    p = counts.astype(np.float64) / float(counts.sum())
    return float(-(p * np.log(p + 1e-12)).sum())


def _aggregate_happo_learning_curves(curves_dir: Path | str | None) -> dict[str, Any]:
    """Mean/sum coordination proxies from on-disk HAPPO learning curves."""
    out: dict[str, Any] = {
        "policy_loss_last_agent_only": True,
        "source": "learning_curves",
    }
    if curves_dir is None:
        return out
    root = Path(curves_dir)
    if not root.is_dir():
        return out
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("fold*_curve.json")):
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(blob, list):
            rows.extend(r for r in blob if isinstance(r, dict))
        elif isinstance(blob, dict):
            rows.append(blob)
    if not rows:
        return out
    for key in _COORD_SCALAR_KEYS:
        vals = [
            float(r[key])
            for r in rows
            if key in r and isinstance(r[key], (int, float)) and np.isfinite(float(r[key]))
        ]
        if not vals:
            continue
        if key == "teamtr_skips":
            out["teamtr_skips_sum"] = float(np.sum(vals))
        else:
            out[f"{key}_mean"] = float(np.mean(vals))
    orders = [r.get("agent_order") for r in rows if r.get("agent_order") is not None]
    ent = _agent_order_entropy(orders)
    if ent is not None:
        out["agent_order_entropy"] = ent
    return out


HAPPO_CHECKPOINT_FORMAT = 2

_HAPPO_CHECKPOINT_META_KEYS = frozenset(
    {
        "format",
        "policy",
        "optimizer",
        "seed",
        "fold_id",
        "run_config_hash",
        "episode",
        "optimizer_steps",
        "resumed_with_fresh_optimizer",
    }
)


def _is_bare_policy_state_dict(blob: Any) -> bool:
    """True when the checkpoint file is a raw ``policy.state_dict()`` (v1)."""
    if not isinstance(blob, dict) or not blob:
        return False
    if _HAPPO_CHECKPOINT_META_KEYS.intersection(blob.keys()):
        return False
    return all(isinstance(k, str) for k in blob.keys())


def _happo_optimizer_state(trainer: Any) -> dict[str, Any] | None:
    actor_opts = getattr(trainer, "actor_opts", None)
    critic_opt = getattr(trainer, "critic_opt", None)
    if actor_opts is not None or critic_opt is not None:
        return {
            "actor_opts": [o.state_dict() for o in actor_opts] if actor_opts else [],
            "critic_opt": critic_opt.state_dict() if critic_opt is not None else None,
        }
    opt = getattr(trainer, "optimizer", None)
    if opt is not None:
        return {"simple": opt.state_dict()}
    return None


def _load_happo_optimizer_state(trainer: Any, state: Mapping[str, Any]) -> bool:
    if "simple" in state:
        opt = getattr(trainer, "optimizer", None)
        if opt is not None:
            opt.load_state_dict(state["simple"])
            return True
        return False
    restored = False
    actor_opts = getattr(trainer, "actor_opts", None)
    critic_opt = getattr(trainer, "critic_opt", None)
    actor_states = state.get("actor_opts")
    if actor_opts is not None and actor_states:
        for opt, sd in zip(actor_opts, actor_states):
            opt.load_state_dict(sd)
            restored = True
    critic_state = state.get("critic_opt")
    if critic_opt is not None and critic_state is not None:
        critic_opt.load_state_dict(critic_state)
        restored = True
    return restored


def _stamp_resumed_with_fresh_optimizer(
    blob: dict[str, Any],
    cfg: Mapping[str, Any],
    trainer: Any | None,
) -> dict[str, Any]:
    out = dict(blob)
    out["resumed_with_fresh_optimizer"] = True
    if isinstance(cfg, dict):
        cfg["resumed_with_fresh_optimizer"] = True
    if trainer is not None:
        trainer.resumed_with_fresh_optimizer = True
    return out


def _happo_checkpoint_payload(
    policy: Any,
    cfg: Mapping[str, Any],
    *,
    seed: int,
    episode: int,
    optimizer_steps: int,
    trainer: Any | None = None,
) -> dict[str, Any]:
    opt_state = _happo_optimizer_state(trainer) if trainer is not None else None
    return {
        "format": HAPPO_CHECKPOINT_FORMAT,
        "policy": policy.state_dict(),
        "optimizer": opt_state,
        "seed": int(seed),
        "fold_id": cfg.get("_fold_id"),
        "run_config_hash": cfg.get("_run_config_hash"),
        "episode": int(episode),
        "optimizer_steps": int(optimizer_steps),
    }


def _save_happo_checkpoint(
    policy: Any,
    cfg: Mapping[str, Any],
    *,
    seed: int,
    episode: int,
    optimizer_steps: int,
    trainer: Any | None = None,
) -> None:
    ckpt_dir = cfg.get("_checkpoint_dir")
    if not ckpt_dir:
        return
    payload = _happo_checkpoint_payload(
        policy,
        cfg,
        seed=seed,
        episode=episode,
        optimizer_steps=optimizer_steps,
        trainer=trainer,
    )
    d = Path(str(ckpt_dir))
    d.mkdir(parents=True, exist_ok=True)
    fold_tag = payload["fold_id"] if payload["fold_id"] is not None else "na"
    torch.save(payload, d / f"fold{fold_tag}_seed{int(seed)}_ep{int(episode):05d}.pt")


def _discover_latest_happo_checkpoint(
    ckpt_dir: Path | str,
    seed: int,
    fold_id: int,
    run_config_hash: str | None,
) -> Path | None:
    """Return the latest HAPPO checkpoint matching one seed/fold/config cell."""
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


def _maybe_resume_happo_checkpoint(
    policy: Any,
    cfg: Mapping[str, Any],
    *,
    trainer: Any | None = None,
) -> dict[str, Any] | None:
    """Load a prior HAPPO fold checkpoint; fail closed on config-hash drift."""
    resume_path = cfg.get("_resume_checkpoint")
    if not resume_path:
        return None
    p = Path(str(resume_path))
    if not p.exists():
        return None
    raw = torch.load(p, map_location="cpu", weights_only=False)
    if _is_bare_policy_state_dict(raw):
        if hasattr(policy, "load_state_dict"):
            policy.load_state_dict(raw)
        return _stamp_resumed_with_fresh_optimizer({}, cfg, trainer)

    blob: dict[str, Any] = dict(raw)
    expected_hash = cfg.get("_run_config_hash")
    stored_hash = blob.get("run_config_hash")
    if (
        expected_hash is not None
        and stored_hash is not None
        and stored_hash != expected_hash
    ):
        raise RuntimeError(
            "happo checkpoint run_config_hash mismatch (checkpoint="
            f"{stored_hash!r} != cfg={expected_hash!r}); "
            "refusing to resume weights trained under a different config"
        )
    state = blob.get("policy")
    if state is not None and hasattr(policy, "load_state_dict"):
        policy.load_state_dict(state)

    opt_state = blob.get("optimizer")
    restored_opt = False
    if opt_state is not None and trainer is not None:
        restored_opt = _load_happo_optimizer_state(trainer, opt_state)
    if not restored_opt:
        return _stamp_resumed_with_fresh_optimizer(blob, cfg, trainer)
    return blob


def _lagged_w_prev_actions(actions: list[torch.Tensor]) -> torch.Tensor:
    if not actions:
        raise ValueError("actions must be non-empty")
    k = int(actions[0].shape[-1])
    device = actions[0].device
    dtype = actions[0].dtype
    rows = [torch.zeros(1, k, device=device, dtype=dtype)]
    for action in actions[:-1]:
        rows.append(action.detach())
    return torch.cat(rows, dim=0)


def _happo_cfg_defaults(cfg: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    out.setdefault("n_assets", 4)
    out.setdefault("n_paths", 4)
    out.setdefault("n_steps", 24)
    out.setdefault("n_strikes", 11)
    out.setdefault("n_maturities", 3)
    out.setdefault("hurst_exponent", 0.1)
    out.setdefault("d_model", 32)
    out.setdefault("d_state", 8)
    out.setdefault("macro_dim", 8)
    out.setdefault("turnover_limit", 0.25)
    out.setdefault("use_gpu", False)
    out.setdefault("train_world", "historical")
    return out


def _build_surfaces(eq_rets: np.ndarray, cfg: Mapping[str, Any], arm: str) -> Any:
    from mascotrl.env.equity_cmdp_bridge import equity_panel_to_cmdp_tensors
    from mascotrl.simulator import get_surface_tensor

    train_world = str(cfg.get("train_world") or "historical").lower().strip()
    if arm == "eq" or train_world == "historical":
        bridge = equity_panel_to_cmdp_tensors(np.asarray(eq_rets, dtype=np.float64))
        return bridge["surfaces"]
    return get_surface_tensor(dict(cfg))


def _train_happo_on_panel(
    eq_rets: np.ndarray,
    cfg: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
) -> tuple[Any, list[float], list[dict[str, Any]]]:
    from mascotrl.env.cmdp_env import CMDPEnv
    from mascotrl.plugins.registry import build_feature_extractor, build_happo_engine
    from mascotrl.plugins.resolve import resolve_plugins

    cfg_local = _happo_cfg_defaults(cfg)
    torch.manual_seed(int(seed))
    from mascotrl.policy.harl_adapter import resolve_use_harl

    use_harl = resolve_use_harl(cfg_local)
    if use_harl:
        # HARL actors are available via src.policy.harl_adapter; the CMDP
        # HAPPOEngine path remains the spectrum default until HARL buffers
        # are wired into HAPPOTrainer. Stamp for provenance.
        cfg_local = dict(cfg_local)
        cfg_local["_harl_requested"] = True
    k = int(cfg_local["n_assets"])
    exec_spread = float(cfg_local.get("execution_spread_bps", 0.0) or 0.0)
    exec_impact = float(cfg_local.get("execution_impact_coef", 0.0) or 0.0)
    if bool(cfg_local.get("cost_in_decision", False)):
        if exec_spread <= 0.0 and exec_impact <= 0.0:
            raise ValueError("cost_in_decision_requires_nonzero_friction")

    surfaces = _build_surfaces(eq_rets, cfg_local, arm)
    d_model = int(cfg_local["d_model"])
    macro_dim = int(cfg_local["macro_dim"])
    plugins = resolve_plugins(cfg_local)
    fe = build_feature_extractor(
        k, d_model, d_state=cfg_local.get("d_state", 8), cfg=cfg_local, plugins=plugins
    )
    policy = build_happo_engine(k, d_model, macro_dim, cfg=cfg_local, plugins=plugins)
    backend = str(getattr(policy, "_projection_backend", "cvxpy"))
    cfg_local["_projection_backend"] = backend
    # Propagate to caller cfg_fold so run_happo_cpcv can stamp the artifact.
    if isinstance(cfg, dict):
        cfg["_projection_backend"] = backend
    env = CMDPEnv(
        surfaces=surfaces,
        feature_extractor=fe,
        policy=policy,
        d_model=d_model,
        macro_dim=macro_dim,
        use_gpu=False,
        execution_spread_bps=exec_spread,
        execution_impact_coef=exec_impact,
    )
    risk_objective = build_risk_objective(dict(cfg_local))
    cmdp_kw = resolve_cmdp_cfg(cfg_local)
    trainer = HAPPOTrainer(
        policy,
        use_compile=False,
        risk_objective=risk_objective,
        cmdp_enabled=bool(cmdp_kw["cmdp_enabled"]),
        cmdp_limit_d=float(cmdp_kw["cmdp_limit_d"]),
        cmdp_kp=float(cmdp_kw["cmdp_kp"]),
        cmdp_ki=float(cmdp_kw["cmdp_ki"]),
        cmdp_kd=float(cmdp_kw["cmdp_kd"]),
    )

    target_steps = int(cfg_local.get("train_env_steps", 0) or 0)
    n_episodes = int(cfg_local.get("train_episodes", 1) or 1)
    max_steps = max(1, int(env.T) - 2)
    if target_steps > 0:
        ep_len = max(1, max_steps)
        n_episodes = max(n_episodes, int(np.ceil(target_steps / ep_len)))
    if bool(cfg_local.get("_happo_toy_fast", False)):
        n_episodes = 1
        max_steps = min(max_steps, 4)

    turnovers: list[float] = []
    learning_curve: list[dict[str, Any]] = []
    total_updates = 0
    start_ep = 0
    resumed = _maybe_resume_happo_checkpoint(policy, cfg_local, trainer=trainer)
    if resumed is not None:
        start_ep = int(resumed.get("episode", 0) or 0)
        total_updates = int(resumed.get("optimizer_steps", 0) or 0)
        # Preserve prior curve rows so a resumed fold does not lose telemetry.
        prior_curve = cfg_local.get("_learning_curve_prior")
        if isinstance(prior_curve, list):
            learning_curve.extend(prior_curve)
    for ep in range(start_ep, max(1, n_episodes)):
        path = ep % max(1, int(cfg_local["n_paths"]))
        obs = env.reset(path=path)
        w_prev = torch.zeros(1, k)
        enriched, macro, deltas, actions, log_probs, values, rewards, dones, raw_actions = (
            [] for _ in range(9)
        )
        for t in range(max_steps):
            if t == 0 or (t + 1) % 50 == 0:
                print(
                    happo_progress_line(
                        "train_step",
                        seed=seed,
                        ep=ep,
                        n_episodes=n_episodes,
                        step=t + 1,
                        max_steps=max_steps,
                    ),
                    flush=True,
                )
            vol_scale = float(obs.info.get("atm_vol", 0.2))
            w, lp, v, w_raw = policy.act_stochastic(
                obs.enriched, obs.macro, w_prev, obs.deltas, vol_scale=vol_scale
            )
            nxt = env.step(w.detach())
            turnovers.append(float((w.detach() - w_prev).abs().sum()))
            enriched.append(obs.enriched.detach())
            macro.append(obs.macro.detach())
            deltas.append(obs.deltas.detach())
            actions.append(w.detach())
            log_probs.append(lp.detach())
            values.append(v.detach().reshape(-1))
            rewards.append(nxt.reward.detach().reshape(-1))
            dones.append(torch.tensor([float(nxt.done)]))
            raw_actions.append(w_raw.detach())
            w_prev = w.detach()
            obs = nxt
            if nxt.done:
                break
        if not rewards:
            continue
        rew_t = torch.cat(rewards, dim=0)
        actions_cat = torch.cat(actions, dim=0)
        w_prev_cat = _lagged_w_prev_actions(actions)
        delta_t = actions_cat - w_prev_cat
        costs = None
        if cmdp_kw.get("cmdp_enabled"):
            costs = build_step_costs(
                rew_t,
                signal=str(cmdp_kw.get("cmdp_cost_signal", "cvar")),
                deltas=delta_t,
                alpha=float(cmdp_kw.get("cmdp_alpha", 0.95)),
            )
        batch = TrainBatch(
            enriched=torch.cat(enriched, dim=0),
            macro=torch.cat(macro, dim=0),
            w_prev=_lagged_w_prev_actions(actions),
            deltas=torch.cat(deltas, dim=0),
            actions=torch.cat(actions, dim=0),
            log_probs=torch.cat(log_probs, dim=0),
            values=torch.cat(values, dim=0),
            rewards=torch.cat(rewards, dim=0),
            dones=torch.cat(dones, dim=0),
            raw_actions=torch.cat(raw_actions, dim=0),
            terminateds=torch.cat(dones, dim=0),
            costs=costs,
        )
        stats = trainer.update(batch, epochs=1)
        total_updates += 1
        row = {"episode": float(ep), "update_idx": float(total_updates), **stats}
        learning_curve.append(row)
        print(
            happo_progress_line(
                "train_ep",
                seed=seed,
                ep=ep,
                n_episodes=n_episodes,
                backend=str(getattr(policy, "_projection_backend", "?")),
            ),
            flush=True,
        )
        ckpt_every = int(cfg_local.get("checkpoint_every_n_episodes", 0) or 0)
        if ckpt_every <= 0 and str(cfg_local.get("protocol_tier") or "").lower() == "narrative":
            ckpt_every = 1
        if ckpt_every > 0 and (ep + 1) % ckpt_every == 0:
            _save_happo_checkpoint(
                policy,
                cfg_local,
                seed=seed,
                episode=ep + 1,
                optimizer_steps=total_updates,
                trainer=trainer,
            )
    _save_happo_checkpoint(
        policy,
        cfg_local,
        seed=seed,
        episode=max(1, n_episodes),
        optimizer_steps=total_updates,
        trainer=trainer,
    )
    return policy, turnovers, learning_curve


def _eval_happo_on_panel(
    eq_rets: np.ndarray,
    test_dates: Sequence[pd.Timestamp],
    cfg: Mapping[str, Any],
    *,
    arm: str,
    policy: Any,
) -> dict[str, Any]:
    from mascotrl.env.cmdp_env import CMDPEnv
    from mascotrl.plugins.registry import build_feature_extractor
    from mascotrl.plugins.resolve import resolve_plugins

    cfg_local = _happo_cfg_defaults(cfg)
    k = int(cfg_local["n_assets"])
    exec_spread = float(cfg_local.get("execution_spread_bps", 0.0) or 0.0)
    exec_impact = float(cfg_local.get("execution_impact_coef", 0.0) or 0.0)
    surfaces = _build_surfaces(eq_rets, cfg_local, arm)
    d_model = int(cfg_local["d_model"])
    macro_dim = int(cfg_local["macro_dim"])
    plugins = resolve_plugins(cfg_local)
    fe = build_feature_extractor(
        k, d_model, d_state=cfg_local.get("d_state", 8), cfg=cfg_local, plugins=plugins
    )
    env = CMDPEnv(
        surfaces=surfaces,
        feature_extractor=fe,
        policy=policy,
        d_model=d_model,
        macro_dim=macro_dim,
        use_gpu=False,
        execution_spread_bps=exec_spread,
        execution_impact_coef=exec_impact,
    )
    obs = env.reset(path=0)
    w_prev = torch.zeros(1, k)
    pnl: dict[str, float] = {}
    date_rows: list[str] = []
    weight_rows: list[list[float]] = []
    turn_rows: list[float] = []
    s_delta_rows: list[float] = []
    s_turn_rows: list[float] = []
    date_i = 0
    for _ in range(max(1, int(env.T) - 2)):
        if date_i >= len(test_dates):
            break
        vol_scale = float(obs.info.get("atm_vol", 0.2))
        act_out = policy.act_stochastic(
            obs.enriched,
            obs.macro,
            w_prev,
            obs.deltas,
            vol_scale=vol_scale,
            return_slacks=True,
        )
        if len(act_out) >= 6:
            w, _, _, _, s_delta, s_turn = act_out[:6]
        else:
            w = act_out[0]
            s_delta = s_turn = None
        nxt = env.step(w.detach())
        ds = str(pd.Timestamp(test_dates[date_i]).date())
        pnl[ds] = float(nxt.reward.item())
        date_rows.append(ds)
        weight_rows.append([float(x) for x in w.detach().reshape(-1).tolist()])
        turn_rows.append(float((w.detach() - w_prev).abs().sum().item()))
        if s_delta is not None:
            s_delta_rows.append(float(torch.as_tensor(s_delta).reshape(-1).mean().item()))
        if s_turn is not None:
            s_turn_rows.append(float(torch.as_tensor(s_turn).reshape(-1).mean().item()))
        w_prev = w.detach()
        obs = nxt
        date_i += 1
        if nxt.done:
            break
    if not weight_rows:
        return _eval_happo_panel_payload(
            pnl=pnl,
            dates=[],
            weights=np.zeros((0, k), dtype=np.float64),
            turnovers=np.zeros(0, dtype=np.float64),
        )
    return _eval_happo_panel_payload(
        pnl=pnl,
        dates=date_rows,
        weights=np.asarray(weight_rows, dtype=np.float64).reshape(len(weight_rows), k),
        turnovers=np.asarray(turn_rows, dtype=np.float64),
        s_delta=np.asarray(s_delta_rows, dtype=np.float64) if s_delta_rows else None,
        s_turn=np.asarray(s_turn_rows, dtype=np.float64) if s_turn_rows else None,
    )


def _happo_fold_runner(
    fold: CPCVFold,
    *,
    dates: Sequence[pd.Timestamp],
    rets: np.ndarray,
    cfg: Mapping[str, Any],
    arm: str,
    seed: int,
    turnovers_out: list[float],
) -> dict[str, float]:
    train_idx = _indices_for_windows(dates, fold.train_windows)
    test_idx = _indices_for_windows(dates, fold.test_windows)
    if train_idx.size == 0 or test_idx.size == 0:
        return {}
    train_rets = np.asarray(rets[train_idx], dtype=np.float64)
    test_rets = np.asarray(rets[test_idx], dtype=np.float64)
    test_dates = [dates[i] for i in test_idx]
    fold_seed = int(seed) + int(fold.fold_id)
    cfg_fold = dict(cfg)
    cfg_fold["_fold_id"] = int(fold.fold_id)
    ckpt_dir = cfg_fold.get("_checkpoint_dir")
    resume_ckpt = (
        _discover_latest_happo_checkpoint(
            str(ckpt_dir),
            fold_seed,
            int(fold.fold_id),
            cfg_fold.get("_run_config_hash"),
        )
        if ckpt_dir
        else None
    )
    if resume_ckpt is None:
        cfg_fold.pop("_resume_checkpoint", None)
    else:
        cfg_fold["_resume_checkpoint"] = str(resume_ckpt)
    policy, train_turns, curve = _train_happo_on_panel(
        train_rets, cfg_fold, arm=arm, seed=fold_seed
    )
    if cfg_fold.get("_projection_backend") is not None and isinstance(cfg, dict):
        cfg["_projection_backend"] = cfg_fold["_projection_backend"]
    turnovers_out.extend(train_turns)
    # Persist learning_curve alongside checkpoints for resume observability.
    curves_dir = cfg_fold.get("_learning_curves_dir")
    if curves_dir and curve:
        try:
            d = Path(str(curves_dir))
            d.mkdir(parents=True, exist_ok=True)
            (d / f"fold{int(fold.fold_id)}_seed{fold_seed}_curve.json").write_text(
                json.dumps(curve, default=str) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
    return _eval_happo_on_panel(test_rets, test_dates, cfg_fold, arm=arm, policy=policy)


def run_happo_cpcv(
    cfg: Mapping[str, Any],
    arm: str,
    *,
    budget: Mapping[str, Any],
    allow_toy_panel: bool = False,
    no_dry_run: bool = False,
    out_dir: Path | str | None = None,
    resume: bool = True,
) -> tuple[dict[str, Any] | None, str | None]:
    """Train/evaluate HAPPO under CPCV for narrative / full-budget cells."""
    try:
        from scripts.run_spectrum_campaign import _toy_research_panel, _try_om_research_panel
        from mascotrl.eval.equity_substrate import load_lake_dyn_hrp_panel, stamp_equity_obs_defaults
    except Exception as exc:  # noqa: BLE001
        return None, f"panel_import_failed: {exc}"

    cfg_local = _happo_cfg_defaults(cfg)
    k = int(cfg_local.get("n_assets", 8) or 8)
    panel = None
    panel_source = "optionmetrics"
    # Parity with H0 / non-HAPPO spectrum: eq historical uses lake sp500_sec + dyn_hrp.
    if (
        str(arm).lower() == "eq"
        and str(cfg_local.get("train_world") or "historical").lower() == "historical"
        and not bool(cfg_local.get("force_om_panel", False))
    ):
        stamp_equity_obs_defaults(cfg_local)
        try:
            dates_l, rets_l, factors_l, meta = load_lake_dyn_hrp_panel(cfg_local, k=k)
            panel = (dates_l, rets_l, factors_l)
            panel_source = str(meta.get("panel_source") or "lake_sp500_sec")
        except Exception as exc:  # noqa: BLE001
            if no_dry_run and not allow_toy_panel:
                return None, f"lake dyn_hrp panel unavailable for eq happo: {exc}"
            cfg_local.setdefault("_panel_load_errors", []).append(str(exc)[:300])
    if panel is None:
        panel = _try_om_research_panel(dict(cfg_local), k)
        panel_source = "optionmetrics"
    if panel is None:
        if no_dry_run and not allow_toy_panel:
            return None, "toy_panel_refused_under_no_dry_run"
        panel = _toy_research_panel(n_days=64, k=k, seed=int(cfg_local.get("seed", 0) or 0))
        panel_source = "toy"

    dates, rets, _factors = panel
    cpcv = CPCVConfig(
        n_splits=int(budget["cpcv_n_splits"]),
        n_test_groups=int(budget["cpcv_n_test_groups"]),
        purge_days=int(cfg_local.get("cpcv_purge_days", 21) or 21),
        embargo_days=int(cfg_local.get("cpcv_embargo_days", 21) or 21),
    )
    if panel_source == "toy":
        cpcv = CPCVConfig(
            n_splits=min(cpcv.n_splits, 2),
            n_test_groups=1,
            purge_days=int(cfg_local.get("cpcv_purge_days", 0) or 0),
            embargo_days=int(cfg_local.get("cpcv_embargo_days", 0) or 0),
        )
        cfg_local["_happo_toy_fast"] = True

    seeds = list(budget.get("seeds") or [int(cfg_local.get("seed", 0) or 0)])
    all_turnovers: list[float] = []
    seed_summaries: list[dict[str, Any]] = []
    seed_fold_aux: list[dict[int, Any]] = []
    resume_enabled = bool(resume and out_dir is not None)
    cpcv_out = Path(out_dir) if out_dir is not None else None
    use_purgedcv = resolve_use_purgedcv(cfg_local)
    if cpcv_out is not None:
        cpcv_out.mkdir(parents=True, exist_ok=True)
        if cfg_local.get("_checkpoint_dir"):
            Path(str(cfg_local["_checkpoint_dir"])).mkdir(parents=True, exist_ok=True)
        curves = cpcv_out / "learning_curves"
        curves.mkdir(parents=True, exist_ok=True)
        cfg_local["_learning_curves_dir"] = str(curves)

    for seed in seeds:
        fold_turnovers: list[float] = []

        def fold_runner(fold: CPCVFold) -> dict[str, Any]:
            return _happo_fold_runner(
                fold,
                dates=dates,
                rets=np.asarray(rets, dtype=np.float64),
                cfg=cfg_local,
                arm=arm,
                seed=int(seed),
                turnovers_out=fold_turnovers,
            )

        from mascotrl.eval.cpcv_lib import run_cpcv_lib

        _cpcv_runner = run_cpcv_lib if use_purgedcv else run_cpcv
        print(
            happo_progress_line(
                "cpcv_train_start",
                seed=int(seed),
                n_splits=cpcv.n_splits,
                n_test_groups=cpcv.n_test_groups,
            ),
            flush=True,
        )
        cpcv_result = _cpcv_runner(
            dates,
            fold_runner,
            cpcv,
            seed=int(seed),
            arm=str(arm),
            resume=resume_enabled,
            out_dir=cpcv_out,
        )
        print(
            happo_progress_line("cpcv_train_done", seed=int(seed)),
            flush=True,
        )
        all_turnovers.extend(fold_turnovers)
        seed_summaries.append(cpcv_result.get("path_summary") or {})
        seed_fold_aux.append(dict(cpcv_result.get("fold_aux") or {}))

    path_summary = seed_summaries[0] if seed_summaries else {}
    cg = collapse_guard(all_turnovers)
    # Reconstruct representative OOS holdings from seed-0 fold aux (path 0).
    path0_aux = _reconstruct_path0_aux_series(
        dates,
        seed_fold_aux[0] if seed_fold_aux else {},
        cpcv,
    )
    coord = _aggregate_happo_learning_curves(cfg_local.get("_learning_curves_dir"))
    art: dict[str, Any] = {
        "eval_protocol": "combinatorial_purged_cv",
        "claim_tier": str(budget.get("claim_tier") or "research"),
        "real_reference_arm_present": True,
        "panel_source": panel_source,
        "path_summary": path_summary,
        "cpcv": {
            "n_folds": cpcv.n_folds(),
            "n_paths": path_summary.get("n_paths"),
            "config": {
                "n_splits": cpcv.n_splits,
                "n_test_groups": cpcv.n_test_groups,
                "purge_days": cpcv.purge_days,
                "embargo_days": cpcv.embargo_days,
            },
        },
        "collapse_guard": cg,
        "seeds": seeds,
        "use_purgedcv": bool(use_purgedcv),
        "train_env_steps": int(cfg_local.get("train_env_steps", 0) or 0),
        "projection_backend": str(cfg_local.get("_projection_backend") or "cvxpy"),
        "projection_k_ceiling": int(PROJECTION_K_CEILING),
        "friction_applied": bool(
            float(cfg_local.get("execution_spread_bps", 0.0) or 0.0) > 0.0
            or float(cfg_local.get("execution_impact_coef", 0.0) or 0.0) > 0.0
        ),
        "paths": {"0": path0_aux},
        "happo_trainer_stats": coord,
        "coordination_proxies": coord,
    }
    if path0_aux.get("weights"):
        art["weights"] = path0_aux["weights"]
        art["turnovers"] = path0_aux.get("turnover") or []
        art["dates"] = path0_aux.get("dates") or []
    # Aggregate QP slacks from fold aux into path-0 for deskorg projection_slacks.
    s_delta_vals: list[float] = []
    s_turn_vals: list[float] = []
    for aux in (seed_fold_aux[0] if seed_fold_aux else {}).values():
        if not isinstance(aux, dict):
            continue
        for rec in aux.values():
            if not isinstance(rec, dict):
                continue
            if "s_delta" in rec and np.isfinite(float(rec["s_delta"])):
                s_delta_vals.append(float(rec["s_delta"]))
            if "s_turn" in rec and np.isfinite(float(rec["s_turn"])):
                s_turn_vals.append(float(rec["s_turn"]))
    if s_delta_vals:
        path0_aux["s_delta"] = s_delta_vals
        art["paths"]["0"] = path0_aux
    if s_turn_vals:
        path0_aux["s_turn"] = s_turn_vals
        art["paths"]["0"] = path0_aux
    # Align returns panel to path-0 length when possible (behaviour return measures).
    try:
        rets_arr = np.asarray(rets, dtype=np.float64)
        n_w = len(path0_aux.get("weights") or [])
        if n_w > 0 and rets_arr.ndim == 2 and rets_arr.shape[0] >= n_w:
            art["panel_returns"] = rets_arr[-n_w:].tolist()
    except (TypeError, ValueError):
        pass
    if panel_source == "toy":
        art["toy_panel"] = True
    return art, None
