"""Load trained bundles and run deterministic inference / OOS rolls.

HAPPO bundles rebuild from ``deploy_config.json`` + ``weights.pt``.
Single-agent bundles require ``rl_backend`` when ``deploy_config.json`` exists
(fail closed). Legacy bundles without that file fall back to sb3 for
ppo-family algos and custom otherwise.
See ``scripts/ship_model.py`` for ONNX export and user-runnable artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from mascotrl.models.registry import ModelCard, load_card, verify_bundle, zoo_root

# Pre-registered exclusion removed once HAPPO bundle rebuild is wired.
HAPPO_OOS_REPLAY_SUPPORTED = True

# Legacy fallback when deploy_config.json is absent entirely.
_PPO_FAMILY_ALGOS = frozenset({"ppo", "cppo"})


class HAPPOInferenceAgent:
    """Minimal HAPPO replay agent from a saved engine state_dict."""

    name = "happo"

    def __init__(self, engine: Any, *, obs_dim: int, action_dim: int):
        self.engine = engine
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.K = int(getattr(engine, "num_assets", action_dim))
        self._d_model = int(getattr(engine, "enriched_dim", max(1, obs_dim // max(self.K, 1))))

    def act(self, obs: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        x = obs.float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        flat = x.reshape(x.shape[0], -1)
        if flat.shape[1] == self.obs_dim and self.obs_dim == self.K * self._d_model:
            enriched = flat.reshape(x.shape[0], self.K, self._d_model)
        else:
            enriched = flat.unsqueeze(1).expand(-1, self.K, -1)[..., : self._d_model]
        macro = torch.zeros(x.shape[0], int(getattr(self.engine, "macro_dim", 8)))
        w_prev = torch.zeros(x.shape[0], self.K)
        deltas = torch.zeros(x.shape[0], self.K)
        w, _, _, _ = self.engine.act_stochastic(
            enriched, macro, w_prev, deltas, vol_scale=0.2
        )
        if deterministic:
            w, _ = self.engine(enriched, macro, w_prev, deltas)
        return w.squeeze(0) if w.shape[0] == 1 else w


def _agent_policy_module(agent: Any):
    for attr in ("actor", "net", "q"):
        mod = getattr(agent, attr, None)
        if isinstance(mod, torch.nn.Module):
            return mod
    return None


def _rebuild_happo_engine(
    card: ModelCard, blob: Mapping[str, Any], *, root: str | Path | None = None
) -> HAPPOInferenceAgent:
    deploy_path = zoo_root(root) / card.model_id / "deploy_config.json"
    deploy: dict[str, Any] = {}
    if deploy_path.is_file():
        deploy = json.loads(deploy_path.read_text(encoding="utf-8"))
    k = int(deploy.get("n_assets") or card.n_assets or card.action_dim)
    d_model = int(deploy.get("d_model", 32))
    macro_dim = int(deploy.get("macro_dim", 8))
    from mascotrl.policy.happo import HAPPOEngine

    engine = HAPPOEngine(
        k,
        enriched_dim=d_model,
        macro_dim=macro_dim,
        turnover_limit=float(deploy.get("turnover_limit", 0.25)),
    )
    state = blob.get("policy") if isinstance(blob.get("policy"), dict) else blob
    engine.load_state_dict(state, strict=False)
    engine.eval()
    return HAPPOInferenceAgent(
        engine, obs_dim=int(card.obs_dim), action_dim=int(card.action_dim)
    )


def load_policy(
    model_id: str,
    *,
    root: str | Path | None = None,
) -> tuple[Any, ModelCard]:
    """Rebuild agent/policy from a verified bundle. Fail closed on hash drift."""
    card = verify_bundle(model_id, root=root)
    wpath = zoo_root(root) / model_id / "weights.pt"
    blob = torch.load(wpath, map_location="cpu", weights_only=False)

    if card.family == "happo":
        return _rebuild_happo_engine(card, blob, root=root), card

    from mascotrl.policy.single_agent import make_single_agent

    deploy: dict[str, Any] = {}
    deploy_path = zoo_root(root) / model_id / "deploy_config.json"
    if deploy_path.is_file():
        deploy = json.loads(deploy_path.read_text(encoding="utf-8"))
        if "rl_backend" not in deploy or not str(deploy.get("rl_backend") or "").strip():
            raise ValueError(
                f"bundle {model_id}: deploy_config.json exists but lacks "
                "rl_backend (fail closed; re-export with an explicit stamp)"
            )
        rl_backend = str(deploy["rl_backend"]).strip().lower()
    else:
        # Legacy bundles with no deploy_config: infer from saved policy keys.
        # Custom PPO/CPPO persist ``actor.*`` / ``critic.*``; SB3 persists
        # ``mlp_extractor.*`` / ``action_net.*``. Prefer keys over algo family
        # defaults so roundtrips do not rebuild the wrong module class.
        policy_sd = blob.get("policy") if isinstance(blob, dict) else None
        keys = set(policy_sd.keys()) if isinstance(policy_sd, dict) else set()
        if any(
            k.startswith("mlp_extractor")
            or k.startswith("action_net")
            or k.startswith("policy.")
            for k in keys
        ):
            rl_backend = "sb3"
        elif keys:
            rl_backend = "custom"
        else:
            algo = str(card.algo or "").lower().strip()
            rl_backend = "sb3" if algo in _PPO_FAMILY_ALGOS else "custom"

    agent = make_single_agent(
        card.algo,
        obs_dim=int(card.obs_dim),
        action_dim=int(card.action_dim),
        rl_backend=rl_backend,
        hidden=int(deploy.get("ppo_hidden", 64) or 64),
    )
    net = _agent_policy_module(agent)
    if net is None or blob.get("policy") is None:
        raise RuntimeError(f"bundle {model_id} has no loadable policy state_dict")
    net.load_state_dict(blob["policy"])
    return agent, card


def act_weights(model_id: str, obs: np.ndarray, *, root: str | Path | None = None) -> np.ndarray:
    """Deterministic L1-normalised weights matching research_alpha_cpcv roll."""
    agent, card = load_policy(model_id, root=root)
    obs_clean = np.nan_to_num(np.asarray(obs, dtype=np.float32).reshape(-1), nan=0.0)
    obs_t = torch.as_tensor(obs_clean, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action = agent.act(obs_t, deterministic=True)
    w = np.nan_to_num(action.detach().cpu().numpy().reshape(-1), nan=0.0)
    denom = float(np.sum(np.abs(w)))
    if denom > 1e-8:
        w = w / denom
    else:
        k = max(int(card.action_dim), 1)
        w = np.full(k, 1.0 / k, dtype=np.float64)
    return w


def roll_oos_with_agent(
    *,
    returns: np.ndarray,
    factors: np.ndarray,
    dates: Sequence[pd.Timestamp],
    idx: np.ndarray,
    agent: Any,
    cfg: Mapping[str, Any],
    friction: Any,
    train_residualizer: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Shared OOS roll used by research_alpha_cpcv and model-zoo backtest.

    Callers must pass a cfg already index-sliced to the supplied OOS window.
    ``train_residualizer`` must be the train-fold frozen fit so OOS betas are
    not refit on the test window (PIT).
    """
    from mascotrl.eval.research_alpha_train import build_research_hist_env
    from mascotrl.eval.yaml_honesty import track_copy

    if train_residualizer is None:
        raise ValueError(
            "roll_oos_with_agent requires train_residualizer (train-frozen "
            "residualizer); refusing to fit betas on the OOS window"
        )
    if idx.size < 3:
        return {}
    sub_rets = returns[idx]
    sub_fac = factors[idx]
    sub_dates = [dates[int(i)] for i in idx]
    cfg_local = track_copy(cfg)
    env = build_research_hist_env(
        sub_rets, sub_fac, cfg_local, residualizer=train_residualizer
    )
    env.friction = friction
    obs, _ = env.reset()
    out: dict[str, dict[str, Any]] = {}
    terminated = False
    truncated = False
    while not (terminated or truncated):
        t_before = int(env.t)
        obs_clean = np.nan_to_num(np.asarray(obs, dtype=np.float32).reshape(-1), nan=0.0)
        obs_t = torch.as_tensor(obs_clean, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = agent.act(obs_t, deterministic=True)
        w = np.nan_to_num(action.detach().cpu().numpy().reshape(-1), nan=0.0)
        denom = float(np.sum(np.abs(w)))
        if denom > 1e-8:
            w = w / denom
        # else: pass through zeros; do not force 1/K
        next_obs, reward, terminated, truncated, info = env.step(w)
        ds = str(pd.Timestamp(sub_dates[t_before]).date())
        g = float(info.get("gross", 0.0))
        c = float(info.get("cost", 0.0))
        b = float(info.get("borrow", 0.0))
        rf = float(info.get("rf", 0.0))
        exec_w = np.asarray(info.get("post_fill_w", w), dtype=np.float64)
        out[ds] = {
            "total_net": g - c - b - rf,
            "residual": float(info.get("residual", reward)),
            "weights": exec_w.tolist(),
            "turnover": float(info.get("turnover", 0.0)),
            "cost": c + b,
            "gross": g,
        }
        obs = next_obs
    return out


def roll_oos(
    model_id: str,
    *,
    returns: np.ndarray,
    factors: np.ndarray,
    dates: Sequence[pd.Timestamp],
    cfg: Mapping[str, Any],
    friction: Any,
    idx: np.ndarray | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """OOS roll from a bundle path. Always stamped non-confirmatory."""
    from mascotrl.eval.residualization import fit_ff4_residualizer, freeze_residualizer

    agent, card = load_policy(model_id, root=root)
    if idx is None:
        idx = np.arange(len(dates), dtype=int)
    idx_arr = np.asarray(idx, dtype=int)
    # Fit betas on the complement of the OOS window when possible; otherwise
    # on the full panel (model-zoo rolls are already stamped non-confirmatory).
    complement = np.setdiff1d(np.arange(len(dates), dtype=int), idx_arr)
    fit_idx = complement if complement.size >= 8 else np.arange(len(dates), dtype=int)
    train_resid = freeze_residualizer(
        fit_ff4_residualizer(
            np.nanmean(np.asarray(returns)[fit_idx], axis=1),
            np.asarray(factors)[fit_idx],
            fold_id=f"zoo_{model_id}",
        ),
        f"zoo_{model_id}",
    )
    pnl = roll_oos_with_agent(
        returns=returns,
        factors=factors,
        dates=dates,
        idx=idx_arr,
        agent=agent,
        cfg=cfg,
        friction=friction,
        train_residualizer=train_resid,
    )
    return {
        "model_id": model_id,
        "card": card.to_dict(),
        "pnl_by_date": pnl,
        "is_confirmatory": False,
    }
