"""Stable-Baselines3 backbone for single-agent PPO/SAC/TD3/DDPG/DQN/RecurrentPPO.

Wraps SB3 algorithms behind the same ``_BaseAgent`` interface used by
``research_alpha_train``. Custom weight heads and post-sample projection are
preserved. HAPPO/MCPG/RRL stay on the custom path; large-scale DQN stays custom
(SB3 DQN is flat Discrete / small product spaces only).
"""
from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.logging_utils import get_logger
from src.policy.single_agent import (
    PPOAgent,
    _BaseAgent,
    _apply_weight_head,
    compute_gae,
    make_single_agent as _make_custom_agent,
)

log = get_logger("mascotrl.policy.sb3")

_SB3_ALGOS = frozenset({"ppo", "sac", "td3", "ddpg", "dqn", "ppo_recurrent"})


class GymnasiumHistoricalEnv(gym.Env):
    """Gymnasium wrapper around ``HistoricalArmEnv`` for SB3 ``model.learn``."""

    metadata = {"render_modes": []}

    def __init__(self, hist_env: Any):
        super().__init__()
        self._env = hist_env
        k = int(getattr(hist_env, "K", 1))
        obs_dim = int(np.prod(getattr(hist_env, "observation_space", None) or (k * 24,)))
        if hasattr(hist_env, "obs_dim"):
            obs_dim = int(hist_env.obs_dim)
        elif hasattr(hist_env, "_obs_dim"):
            obs_dim = int(hist_env._obs_dim)
        self._obs_dim = obs_dim
        self._k = k
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(k,), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self._env.reset(seed=seed)
        return np.nan_to_num(np.asarray(obs, dtype=np.float32).reshape(-1), nan=0.0), info

    def step(self, action):
        w = np.asarray(action, dtype=np.float64).reshape(-1)
        denom = float(np.sum(np.abs(w)))
        if denom > 1e-8:
            w = w / denom
        # else: keep zero vector (intentional flat / no exposure)
        obs, reward, terminated, truncated, info = self._env.step(w)
        obs_clean = np.nan_to_num(np.asarray(obs, dtype=np.float32).reshape(-1), nan=0.0)
        return obs_clean, float(reward), bool(terminated), bool(truncated), info


class _SB3PolicyMixin:
    """Shared act / weight-head helpers for SB3-backed agents."""

    backend = "sb3"
    weight_head: str = "softmax"
    weight_head_temperature: float = 1.0
    weight_head_tilt_gain: float = 1.0

    @property
    def net(self) -> nn.Module:
        """Expose SB3 policy module for research checkpoint save/resume."""
        return self._model.policy  # type: ignore[attr-defined]

    @property
    def opt(self) -> Any:
        """Expose SB3 policy optimizer for checkpoint save/resume."""
        return getattr(self._model.policy, "optimizer", None)  # type: ignore[attr-defined]

    def raw_to_weights(self, raw: torch.Tensor) -> torch.Tensor:
        return _apply_weight_head(
            raw,
            self.weight_head,
            temperature=self.weight_head_temperature,
            tilt_gain=self.weight_head_tilt_gain,
        )

    def act(self, obs: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        x = obs.detach().cpu().numpy()
        if x.ndim == 1:
            x = x.reshape(1, -1)
        action, _ = self._model.predict(x, deterministic=deterministic)
        raw = torch.as_tensor(action, dtype=torch.float32)
        if raw.ndim == 1:
            raw = raw.unsqueeze(0)
        return self.raw_to_weights(raw)

    def act_and_logp_raw(
        self, obs: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._prep_obs(obs)
        dist = self._model.policy.get_distribution(x)
        if deterministic:
            raw = dist.mode()
        else:
            raw = dist.sample()
        logp = dist.log_prob(raw)
        if logp.ndim > 1:
            logp = logp.sum(dim=-1)
        return raw, logp

    def _prep_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.float()


class SB3PPOAgent(_SB3PolicyMixin, _BaseAgent):
    name = "ppo"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.02,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        weight_head: str = "softmax",
        weight_head_temperature: float = 1.0,
        weight_head_tilt_gain: float = 1.0,
        **kwargs: Any,
    ):
        from stable_baselines3 import PPO

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.clip_eps = float(clip_eps)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.weight_head = str(weight_head or "softmax").lower()
        self.weight_head_temperature = float(weight_head_temperature)
        self.weight_head_tilt_gain = float(weight_head_tilt_gain)
        self.normalize_adv = True
        self._optimizer_steps = 0

        class _Dummy(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self, od: int, ad: int):
                super().__init__()
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(od,), dtype=np.float32
                )
                self.action_space = gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(ad,), dtype=np.float32
                )

            def reset(self, *, seed=None, options=None):
                return np.zeros(self.observation_space.shape, dtype=np.float32), {}

            def step(self, action):
                return (
                    np.zeros(self.observation_space.shape, dtype=np.float32),
                    0.0,
                    True,
                    False,
                    {},
                )

        dummy = _Dummy(obs_dim, action_dim)
        self._model = PPO(
            "MlpPolicy",
            dummy,
            learning_rate=lr,
            n_steps=64,
            batch_size=64,
            n_epochs=1,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_eps,
            ent_coef=entropy_coef,
            vf_coef=value_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs={"net_arch": [64, 64]},
            verbose=0,
        )

    def train_epoch(
        self,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        old_logprobs: torch.Tensor | None = None,
        n_epochs: int = 1,
        n_minibatches: int = 1,
        sample_weight: torch.Tensor | None = None,
        policy_step_mask: torch.Tensor | None = None,
        scr_mix: str = "off",
        scr_beta: float = 0.5,
        scr_y_cf: torch.Tensor | None = None,
    ) -> dict[str, float]:
        policy = self._model.policy
        obs_f = self._prep_obs(obs)
        with torch.no_grad():
            values = policy.predict_values(obs_f).flatten()
            next_values = policy.predict_values(self._prep_obs(next_obs)).flatten()
            from src.eval.scr_critic import build_scr_returns

            advantages, returns, _scr_meta = build_scr_returns(
                rewards=rewards,
                values=values,
                next_values=next_values,
                dones=dones,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                scr_mix=str(scr_mix or "off"),
                scr_beta=float(scr_beta),
                y_cf=scr_y_cf,
            )
            # RC5: episode_weights before z-norm; optional rebalance mask.
            # Inactive steps stay at 0 after norm (no -mean/std leak).
            active = torch.ones_like(advantages, dtype=torch.bool)
            if sample_weight is not None:
                sw = sample_weight.detach().reshape(-1).to(advantages.dtype)
                advantages = advantages * sw
                active = active & (sw.abs() > 0)
            if policy_step_mask is not None:
                m = policy_step_mask.detach().reshape(-1).to(dtype=torch.bool)
                if m.shape[0] != advantages.shape[0]:
                    raise ValueError(
                        f"policy_step_mask length {m.shape[0]} != batch {advantages.shape[0]}"
                    )
                active = active & m
            advantages = torch.where(active, advantages, torch.zeros_like(advantages))
            if self.normalize_adv:
                if bool(active.any().item()):
                    act = advantages[active]
                    mu = act.mean()
                    sd = act.std(unbiased=False) + 1e-8
                    advantages = torch.where(
                        active, (advantages - mu) / sd, torch.zeros_like(advantages)
                    )
            if old_logprobs is None:
                _, old_logprobs, _ = policy.evaluate_actions(obs_f, actions)
            else:
                old_logprobs = old_logprobs.detach()

        n = int(obs.shape[0])
        mb = max(1, int(n_minibatches))
        batch_size = max(1, n // mb)
        last: dict[str, float] = {}
        clip_fracs: list[float] = []
        kl_vals: list[float] = []
        entropies: list[float] = []
        p_gn = 0.0
        v_gn = 0.0

        for _ in range(max(1, int(n_epochs))):
            perm = torch.randperm(n)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                if idx.numel() == 0:
                    continue
                xb = self._prep_obs(obs[idx])
                act_b = actions[idx]
                _, new_logp, entropy = policy.evaluate_actions(xb, act_b)
                ratio = torch.exp(new_logp.flatten() - old_logprobs[idx].flatten())
                adv = advantages[idx]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_pred = policy.predict_values(xb).flatten()
                value_loss = F.mse_loss(value_pred, returns[idx])
                ent = entropy.mean() if entropy.ndim else entropy
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * ent
                )
                policy.optimizer.zero_grad()
                loss.backward()
                p_gn = max(
                    p_gn,
                    float(
                        nn.utils.clip_grad_norm_(
                            policy.parameters(), self.max_grad_norm
                        ).item()
                    ),
                )
                policy.optimizer.step()
                clip_fracs.append(
                    float((torch.abs(ratio - 1.0) > self.clip_eps).float().mean())
                )
                kl_vals.append(
                    float((old_logprobs[idx] - new_logp.flatten()).detach().mean())
                )
                entropies.append(float(ent.detach()))
                self._optimizer_steps += 1
                last = {
                    "loss": float(loss.detach()),
                    "actor_loss": float(policy_loss.detach()),
                    "critic_loss": float(value_loss.detach()),
                    "entropy": float(ent.detach()),
                    "approx_kl": float(np.mean(kl_vals)) if kl_vals else 0.0,
                    "clip_frac": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
                    "grad_norm": p_gn,
                    "policy_grad_norm": p_gn,
                    "value_grad_norm": p_gn,
                    # Per-call steps (not cumulative) so research_alpha_train
                    # can sum without double-counting across folds.
                    "optimizer_steps": float(len(clip_fracs)),
                    "rl_backend": "sb3",
                }
        return last


class SB3OffPolicyAgent(_SB3PolicyMixin, _BaseAgent):
    """SAC/TD3/DDPG via SB3 with replay-buffer train_epoch."""

    def __init__(
        self,
        algo: str,
        obs_dim: int,
        action_dim: int,
        *,
        lr: float = 3e-4,
        gamma: float = 0.99,
        weight_head: str = "tanh_l1",
        **kwargs: Any,
    ):
        from stable_baselines3 import DDPG, SAC, TD3

        cls_map = {"sac": SAC, "td3": TD3, "ddpg": DDPG}
        sb3_cls = cls_map[str(algo).lower()]
        self.name = str(algo).lower()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.weight_head = str(weight_head or "tanh_l1").lower()
        self.weight_head_temperature = float(kwargs.get("weight_head_temperature", 1.0))
        self.weight_head_tilt_gain = float(kwargs.get("weight_head_tilt_gain", 1.0))
        self._optimizer_steps = 0

        class _Dummy(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self, od: int, ad: int):
                super().__init__()
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(od,), dtype=np.float32
                )
                self.action_space = gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(ad,), dtype=np.float32
                )

            def reset(self, *, seed=None, options=None):
                return np.zeros(self.observation_space.shape, dtype=np.float32), {}

            def step(self, action):
                return (
                    np.zeros(self.observation_space.shape, dtype=np.float32),
                    0.0,
                    True,
                    False,
                    {},
                )

        dummy = _Dummy(obs_dim, action_dim)
        self._model = sb3_cls(
            "MlpPolicy",
            dummy,
            learning_rate=lr,
            gamma=gamma,
            policy_kwargs={"net_arch": [64, 64]},
            verbose=0,
            buffer_size=max(1000, obs_dim * 4),
            learning_starts=32,
            batch_size=32,
        )
        # OffPolicyAlgorithm.train() needs _logger + buffer metadata from learn setup.
        self._model._setup_learn(total_timesteps=0)
        self._replay: list[tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, float]] = []

    def act_and_logp_raw(
        self, obs: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample pre-head actions for the research collect path.

        SAC exposes ``actor.action_log_prob``; TD3/DDPG are deterministic
        actors (logp stamped 0). Do not call ``policy.get_distribution`` —
        that API is on-policy (PPO) only.
        """
        x = self._prep_obs(obs)
        policy = self._model.policy
        actor = getattr(policy, "actor", None)
        if actor is not None and hasattr(actor, "action_log_prob"):
            # SAC: returns (actions, log_prob)
            raw, logp = actor.action_log_prob(x)
            raw = torch.as_tensor(raw, dtype=torch.float32)
            logp = torch.as_tensor(logp, dtype=torch.float32).reshape(-1)
            if deterministic and hasattr(actor, "get_action_dist_params"):
                mean, _log_std, _kwargs = actor.get_action_dist_params(x)
                raw = torch.as_tensor(mean, dtype=torch.float32)
            if raw.ndim == 1:
                raw = raw.unsqueeze(0)
            return raw, logp
        # TD3 / DDPG: deterministic mu(s); exploration noise is optional.
        if actor is not None:
            raw = actor(x)
            raw = torch.as_tensor(raw, dtype=torch.float32)
            if raw.ndim == 1:
                raw = raw.unsqueeze(0)
            if not deterministic:
                raw = raw + 0.1 * torch.randn_like(raw)
                raw = torch.clamp(raw, -1.0, 1.0)
            logp = torch.zeros(raw.shape[0], dtype=torch.float32)
            return raw, logp
        # Last resort: predict path (weights already projected by act()).
        action, _ = self._model.predict(x.detach().cpu().numpy(), deterministic=deterministic)
        raw = torch.as_tensor(action, dtype=torch.float32)
        if raw.ndim == 1:
            raw = raw.unsqueeze(0)
        return raw, torch.zeros(raw.shape[0], dtype=torch.float32)

    def train_epoch(
        self,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        for t in range(int(obs.shape[0])):
            self._model.replay_buffer.add(
                obs[t].detach().cpu().numpy(),
                next_obs[t].detach().cpu().numpy(),
                actions[t].detach().cpu().numpy(),
                float(rewards[t].item()),
                bool(dones[t].item() > 0.5),
                [{}],
            )
        if self._model.replay_buffer.size() < self._model.learning_starts:
            return {
                "loss": 0.0,
                "rl_backend": "sb3",
                "optimizer_steps": 0.0,
                "loss_source": "stub",
            }
        self._model.train(batch_size=self._model.batch_size, gradient_steps=1)
        self._optimizer_steps += 1
        loss_val = 0.0
        loss_source = "stub"
        try:
            n2v = getattr(getattr(self._model, "logger", None), "name_to_value", None) or {}
            for k in ("train/loss", "train/actor_loss", "train/critic_loss", "loss"):
                if k in n2v:
                    loss_val = float(n2v[k])
                    loss_source = "sb3_logger"
                    break
        except Exception:
            pass
        return {
            "loss": float(loss_val),
            "rl_backend": "sb3",
            "optimizer_steps": 1.0,
            "loss_source": loss_source,
        }


class PortfolioFeaturesExtractor:
    """SB3 features extractor: reshape flat ``(B, K*seq*C)`` then MLP embed."""

    def __init__(
        self,
        *,
        features_dim: int = 64,
        num_assets: int = 1,
        seq_len: int = 1,
        n_channels: int | None = None,
    ):
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

        class _Extractor(BaseFeaturesExtractor):
            def __init__(self, obs_space):
                flat = int(np.prod(obs_space.shape))
                super().__init__(obs_space, features_dim=int(features_dim))
                self._num_assets = int(num_assets)
                self._seq_len = int(seq_len)
                c = n_channels
                if c is None:
                    denom = max(self._num_assets * self._seq_len, 1)
                    c = max(1, flat // denom)
                self._n_channels = int(c)
                self._net = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(flat, int(features_dim)),
                    nn.ReLU(),
                    nn.Linear(int(features_dim), int(features_dim)),
                    nn.ReLU(),
                )

            def forward(self, observations):
                x = observations.float()
                if x.ndim == 1:
                    x = x.unsqueeze(0)
                b = x.shape[0]
                expected = self._num_assets * self._seq_len * self._n_channels
                if x.shape[-1] == expected:
                    x = x.view(b, self._num_assets, self._seq_len, self._n_channels)
                    x = x.reshape(b, -1)
                return self._net(x)

        self.cls = _Extractor


class SB3DQNAgent(_SB3PolicyMixin, _BaseAgent):
    """Per-asset discrete Q via flattened Discrete (product of bins).

    SB3 DQN only accepts ``Discrete`` (not MultiDiscrete). For small ``K`` we
    encode the MultiDiscrete as a single Discrete of size ``n_bins**K``.
    Large ``K`` must stay on the custom DQN path.
    """

    name = "dqn"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        lr: float = 3e-4,
        gamma: float = 0.99,
        n_bins: int = 3,
        **kwargs: Any,
    ):
        from stable_baselines3 import DQN

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.n_bins = int(n_bins)
        self.gamma = float(gamma)
        self.weight_head = "discrete"
        self.weight_head_temperature = 1.0
        self.weight_head_tilt_gain = 1.0
        self._optimizer_steps = 0
        n_flat = int(self.n_bins**self.action_dim)
        if n_flat > 512:
            raise ValueError(
                f"SB3 DQN product space n_bins**K={n_flat} > 512; use custom dqn"
            )

        class _Dummy(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self, od: int, n_act: int):
                super().__init__()
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(od,), dtype=np.float32
                )
                self.action_space = gym.spaces.Discrete(n_act)

            def reset(self, *, seed=None, options=None):
                return np.zeros(self.observation_space.shape, dtype=np.float32), {}

            def step(self, action):
                return (
                    np.zeros(self.observation_space.shape, dtype=np.float32),
                    0.0,
                    True,
                    False,
                    {},
                )

        dummy = _Dummy(obs_dim, n_flat)
        self._n_flat = n_flat
        self._model = DQN(
            "MlpPolicy",
            dummy,
            learning_rate=lr,
            gamma=gamma,
            policy_kwargs={"net_arch": [64, 64]},
            verbose=0,
            buffer_size=max(1000, obs_dim * 4),
            learning_starts=32,
            batch_size=32,
        )
        # OffPolicyAlgorithm.train() needs _logger + buffer metadata from learn setup.
        self._model._setup_learn(total_timesteps=0)

    def _decode(self, flat: int) -> np.ndarray:
        out = np.zeros(self.action_dim, dtype=np.float64)
        x = int(flat)
        for i in range(self.action_dim - 1, -1, -1):
            out[i] = x % self.n_bins
            x //= self.n_bins
        return out

    def _encode(self, bins: np.ndarray) -> int:
        x = 0
        for b in bins.astype(int).tolist():
            x = x * self.n_bins + int(b)
        return int(x)

    def _bins_to_levels(self, bins: np.ndarray) -> np.ndarray:
        """Map bin indices {0,...,n_bins-1} to centered levels (custom DQN parity).

        For ``n_bins=3`` this yields ``{-1, 0, 1}`` so long-short L1 portfolios
        are expressible (not a long-only simplex).
        """
        center = (self.n_bins - 1) / 2.0
        return bins.astype(np.float64) - center

    def act(self, obs: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        x = obs.detach().cpu().numpy()
        if x.ndim == 1:
            x = x.reshape(1, -1)
        action, _ = self._model.predict(x, deterministic=deterministic)
        flats = np.asarray(action).reshape(-1)
        # SB3 VectorizedEnv predict may return one action per env; batch obs
        # through a DummyVecEnv of size 1 still yields a single flat index when
        # we call predict on a (B, obs) array — decode each row independently.
        if flats.size == 1 and x.shape[0] > 1:
            # predict collapsed batch; re-run per row for batch semantics
            rows = []
            for i in range(x.shape[0]):
                a_i, _ = self._model.predict(x[i : i + 1], deterministic=deterministic)
                rows.append(self._decode(int(np.asarray(a_i).reshape(-1)[0])))
            bins_batch = np.stack(rows, axis=0)
        else:
            bins_batch = np.stack([self._decode(int(f)) for f in flats], axis=0)
        levels = self._bins_to_levels(bins_batch)
        denom = np.abs(levels).sum(axis=-1, keepdims=True)
        denom = np.maximum(denom, 1e-8)
        w = levels / denom
        # Zero exposure when all levels are zero (denom was floored).
        zero_mask = np.abs(levels).sum(axis=-1) < 1e-8
        w[zero_mask] = 0.0
        return torch.as_tensor(w, dtype=torch.float32)

    @property
    def act_and_logp_raw(self):  # type: ignore[override]
        # Mixin method calls policy.get_distribution (PPO-only); raise so
        # hasattr is False and research_alpha_train falls back to act().
        raise AttributeError(
            "SB3DQNAgent has no act_and_logp_raw; collect via act()"
        )

    def train_epoch(
        self,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        for t in range(int(obs.shape[0])):
            act = actions[t].detach().cpu().numpy().reshape(-1)
            if act.size == 1:
                discrete = int(act[0]) % self._n_flat
            elif act.size == self.action_dim:
                span = float(np.ptp(act)) + 1e-8
                bins = np.clip(
                    np.round((act - float(np.min(act))) / span * (self.n_bins - 1)),
                    0,
                    self.n_bins - 1,
                )
                discrete = self._encode(bins)
            else:
                discrete = 0
            self._model.replay_buffer.add(
                obs[t].detach().cpu().numpy(),
                next_obs[t].detach().cpu().numpy(),
                np.array([discrete]),
                float(rewards[t].item()),
                bool(dones[t].item() > 0.5),
                [{}],
            )
        if self._model.replay_buffer.size() < self._model.learning_starts:
            return {
                "loss": 0.0,
                "rl_backend": "sb3",
                "optimizer_steps": 0.0,
                "loss_source": "stub",
            }
        self._model.train(batch_size=self._model.batch_size, gradient_steps=1)
        self._optimizer_steps += 1
        loss_val = 0.0
        loss_source = "stub"
        try:
            n2v = getattr(getattr(self._model, "logger", None), "name_to_value", None) or {}
            for k in ("train/loss", "train/actor_loss", "train/critic_loss", "loss"):
                if k in n2v:
                    loss_val = float(n2v[k])
                    loss_source = "sb3_logger"
                    break
        except Exception:
            pass
        return {
            "loss": float(loss_val),
            "rl_backend": "sb3",
            "optimizer_steps": 1.0,
            "loss_source": loss_source,
        }


class SB3RecurrentPPOAgent(_SB3PolicyMixin, _BaseAgent):
    """sb3-contrib RecurrentPPO with portfolio feature extractor."""

    name = "ppo_recurrent"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        lr: float = 3e-4,
        gamma: float = 0.99,
        weight_head: str = "softmax",
        num_assets: int = 1,
        seq_len: int = 1,
        **kwargs: Any,
    ):
        from sb3_contrib import RecurrentPPO

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.weight_head = str(weight_head or "softmax").lower()
        self.weight_head_temperature = float(kwargs.get("weight_head_temperature", 1.0))
        self.weight_head_tilt_gain = float(kwargs.get("weight_head_tilt_gain", 1.0))
        self._optimizer_steps = 0
        extractor = PortfolioFeaturesExtractor(
            features_dim=64,
            num_assets=int(num_assets),
            seq_len=int(seq_len),
        )

        class _Dummy(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self, od: int, ad: int):
                super().__init__()
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(od,), dtype=np.float32
                )
                self.action_space = gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(ad,), dtype=np.float32
                )

            def reset(self, *, seed=None, options=None):
                return np.zeros(self.observation_space.shape, dtype=np.float32), {}

            def step(self, action):
                return (
                    np.zeros(self.observation_space.shape, dtype=np.float32),
                    0.0,
                    True,
                    False,
                    {},
                )

        dummy = _Dummy(obs_dim, action_dim)
        self._model = RecurrentPPO(
            "MlpLstmPolicy",
            dummy,
            learning_rate=lr,
            gamma=gamma,
            n_steps=64,
            batch_size=64,
            n_epochs=1,
            policy_kwargs={
                "features_extractor_class": extractor.cls,
                "net_arch": [64],
                "lstm_hidden_size": 64,
            },
            verbose=0,
        )

    @property
    def act_and_logp_raw(self):  # type: ignore[override]
        # Mixin method needs lstm_states/episode_starts; raise so hasattr is
        # False and research_alpha_train falls back to act() (logp=0).
        raise AttributeError(
            "SB3RecurrentPPOAgent has no act_and_logp_raw; collect via act()"
        )

    def train_epoch(
        self,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        raise NotImplementedError(
            "SB3 RecurrentPPO train_epoch is not implemented; "
            "use rl_backend=custom for PPO+gru/lstm"
        )


def make_sb3_agent(
    algo: str,
    *,
    obs_dim: int,
    action_dim: int,
    **kwargs: Any,
) -> _BaseAgent:
    key = str(algo).lower().strip()
    if key not in _SB3_ALGOS:
        raise KeyError(f"SB3 backend does not support algo={algo!r}")
    if key == "ppo":
        return SB3PPOAgent(obs_dim, action_dim, **kwargs)
    if key == "dqn":
        return SB3DQNAgent(obs_dim, action_dim, **kwargs)
    if key == "ppo_recurrent":
        return SB3RecurrentPPOAgent(obs_dim, action_dim, **kwargs)
    return SB3OffPolicyAgent(key, obs_dim, action_dim, **kwargs)


def resolve_rl_backend(cfg: Mapping[str, Any] | None) -> str:
    if cfg is None:
        return "sb3"
    return str(cfg.get("rl_backend") or "sb3").lower().strip()


def make_single_agent_with_backend(
    algo: str,
    *,
    obs_dim: int,
    action_dim: int,
    cfg: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> _BaseAgent:
    """Route to SB3 or custom implementation based on ``rl_backend`` YAML key."""
    key = str(algo).lower().strip()
    backend = resolve_rl_backend(cfg)
    if backend == "sb3" and key in _SB3_ALGOS:
        try:
            return make_sb3_agent(key, obs_dim=obs_dim, action_dim=action_dim, **kwargs)
        except ImportError as exc:
            log.warning("SB3 import failed; falling back to custom: %s", exc)
    return _make_custom_agent(key, obs_dim=obs_dim, action_dim=action_dim, **kwargs)
