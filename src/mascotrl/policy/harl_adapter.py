"""HARL HAPPO adapter: multi-agent wrapper around HistoricalArmEnv.

Sequential per-agent updates; weight-head + cvxpy projection stay in MascotRL.
Enable via cell YAML ``use_harl: true``.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from mascotrl.logging_utils import get_logger

log = get_logger("mascotrl.policy.harl")

_DEFAULT_HAPPO_ARGS: dict[str, Any] = {
    "hidden_sizes": [64, 64],
    "gain": 0.01,
    "use_orthogonal": True,
    "use_policy_active_masks": True,
    "use_naive_recurrent_policy": False,
    "use_recurrent_policy": False,
    "recurrent_n": 1,
    "data_chunk_length": 10,
    "lr": 5e-4,
    "opti_eps": 1e-5,
    "weight_decay": 0.0,
    "clip_param": 0.2,
    "ppo_epoch": 1,
    "actor_num_mini_batch": 1,
    "entropy_coef": 0.01,
    "use_max_grad_norm": True,
    "max_grad_norm": 10.0,
    "use_clipped_value_loss": True,
    "use_valuenorm": False,
    "use_feature_normalization": False,
    "use_ReLU": True,
    "activation_func": "relu",
    "initialization_method": "orthogonal_",
    "layered_state": False,
    "std_x_coef": 1.0,
    "std_y_coef": 0.5,
    "action_aggregation": "prod",
    "use_policy_vtrace": False,
}


def default_happo_args(**overrides: Any) -> dict[str, Any]:
    out = dict(_DEFAULT_HAPPO_ARGS)
    out.update(overrides)
    return out


class HistoricalArmHARLEnv:
    """Multi-agent view of ``HistoricalArmEnv`` for HARL's reset/step API.

    Each asset is one agent. Observations are per-asset slices of the flat
    feature vector when possible; otherwise the full obs is shared.
    Actions are raw Δw scalars; the caller applies weight-head + projection.
    """

    def __init__(self, hist_env: Any, *, n_agents: int | None = None):
        self._env = hist_env
        k = int(n_agents or getattr(hist_env, "K", 1))
        self.n_agents = k
        obs_dim = int(getattr(hist_env, "obs_dim", None) or getattr(hist_env, "_obs_dim", k * 8))
        self._obs_dim = obs_dim
        self._per_agent_dim = max(1, obs_dim // k)
        import gymnasium as gym

        self.observation_space = [
            gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._per_agent_dim,),
                dtype=np.float32,
            )
            for _ in range(k)
        ]
        self.share_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = [
            gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            for _ in range(k)
        ]

    def _split_obs(self, obs: np.ndarray) -> list[np.ndarray]:
        flat = np.nan_to_num(np.asarray(obs, dtype=np.float32).reshape(-1), nan=0.0)
        if flat.size < self.n_agents * self._per_agent_dim:
            flat = np.pad(flat, (0, self.n_agents * self._per_agent_dim - flat.size))
        out = []
        for i in range(self.n_agents):
            sl = flat[i * self._per_agent_dim : (i + 1) * self._per_agent_dim]
            out.append(sl.astype(np.float32))
        return out

    def reset(self, *, seed: int | None = None):
        obs, info = self._env.reset(seed=seed)
        flat = np.nan_to_num(np.asarray(obs, dtype=np.float32).reshape(-1), nan=0.0)
        agent_obs = self._split_obs(flat)
        share = flat[: self._obs_dim].astype(np.float32)
        if share.size < self._obs_dim:
            share = np.pad(share, (0, self._obs_dim - share.size))
        available = [None] * self.n_agents
        return agent_obs, share, available

    def step(self, actions: Sequence[np.ndarray]):
        raw = np.asarray([float(np.asarray(a).reshape(-1)[0]) for a in actions], dtype=np.float64)
        denom = float(np.sum(np.abs(raw)))
        if denom > 1e-8:
            w = raw / denom
        else:
            w = np.full(self.n_agents, 1.0 / max(self.n_agents, 1), dtype=np.float64)
        obs, reward, terminated, truncated, info = self._env.step(w)
        flat = np.nan_to_num(np.asarray(obs, dtype=np.float32).reshape(-1), nan=0.0)
        agent_obs = self._split_obs(flat)
        share = flat[: self._obs_dim].astype(np.float32)
        if share.size < self._obs_dim:
            share = np.pad(share, (0, self._obs_dim - share.size))
        rewards = [[float(reward)] for _ in range(self.n_agents)]
        dones = [bool(terminated or truncated)] * self.n_agents
        infos = [dict(info) for _ in range(self.n_agents)]
        available = [None] * self.n_agents
        return agent_obs, share, rewards, dones, infos, available


class HARLHAPPOBundle:
    """K HARL HAPPO actors updated in sequential agent order."""

    def __init__(
        self,
        n_agents: int,
        obs_dim_per_agent: int,
        *,
        args: Mapping[str, Any] | None = None,
        device: str = "cpu",
    ):
        import gymnasium as gym
        import torch
        from harl.algorithms.actors.happo import HAPPO

        self.n_agents = int(n_agents)
        self.args = default_happo_args(**dict(args or {}))
        self.device = torch.device(device)
        obs_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(obs_dim_per_agent),),
            dtype=np.float32,
        )
        act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.actors = [
            HAPPO(self.args, obs_space, act_space, device=self.device)
            for _ in range(self.n_agents)
        ]

    def sequential_agent_order(self, *, seed: int | None = None) -> list[int]:
        """Return agent update order; optional shuffle still visits each once."""
        order = list(range(self.n_agents))
        if seed is not None:
            rng = np.random.default_rng(int(seed))
            rng.shuffle(order)
        return order

    def act_all(
        self,
        obs_list: Sequence[np.ndarray],
        *,
        deterministic: bool = True,
    ) -> list[np.ndarray]:
        """Sample one action per agent (zero RNN state for MLP policies)."""
        hidden = int(self.args["hidden_sizes"][-1])
        recurrent_n = int(self.args.get("recurrent_n", 1) or 1)
        actions: list[np.ndarray] = []
        for i, actor in enumerate(self.actors):
            o = np.asarray(obs_list[i], dtype=np.float32).reshape(1, -1)
            rnn = np.zeros((1, recurrent_n, hidden), dtype=np.float32)
            masks = np.ones((1, 1), dtype=np.float32)
            out = actor.act(
                o, rnn, masks, available_actions=None, deterministic=deterministic
            )
            a = out[0] if isinstance(out, (tuple, list)) else out
            if hasattr(a, "detach"):
                a = a.detach().cpu().numpy()
            actions.append(np.asarray(a, dtype=np.float32).reshape(-1))
        return actions


def resolve_use_harl(cfg: Mapping[str, Any] | None) -> bool:
    if cfg is None:
        return False
    return bool(cfg.get("use_harl", False))
