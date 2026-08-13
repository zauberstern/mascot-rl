"""Shared networks, weight heads, and base types for single-agent RL adapters."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mascotrl.policy.bodies import (
    AssetTemporalPolicyBody,
    build_policy_body,
)

def _mlp(in_dim: int, out_dim: int, hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


def _actor_body(
    *,
    architecture: str,
    obs_dim: int,
    action_dim: int,
    hidden: int = 64,
    num_assets: int | None = None,
    d_model: int | None = None,
    seq_len: int = 1,
    d_state: int = 16,
    share_temporal_encoder: bool = True,
    use_surface_image_encoder: bool = False,
    image_channels: int = 0,
    surface_image_embed_dim: int = 16,
    with_critic: bool = False,
    actor_final_gain: float = 0.1,
) -> nn.Module:
    """Shared actor trunk for PPO / SAC / TD3 / DDPG."""
    spec = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden": hidden,
        "num_assets": num_assets,
        "d_model": d_model,
        "seq_len": seq_len,
        "d_state": d_state,
        "share_temporal_encoder": share_temporal_encoder,
        "use_surface_image_encoder": use_surface_image_encoder,
        "image_channels": image_channels,
        "surface_image_embed_dim": surface_image_embed_dim,
        "with_critic": with_critic,
        "actor_final_gain": actor_final_gain,
    }
    return build_policy_body(architecture, spec, {})


_WEIGHT_HEADS = frozenset(
    {
        "softmax",
        "tanh_l1",
        "raw",
        "sparse_tilt",
        "sparse_tilt_tsallis",
        "entmax_15",
        "dirichlet_tilt",
        "dirichlet_mean",
        "dirichlet_entropy",
    }
)


def _apply_weight_head(
    raw: torch.Tensor,
    head: str,
    *,
    temperature: float = 1.0,
    tilt_gain: float = 1.0,
    w_base: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map actor output to portfolio weights.

    ``softmax``: long-only simplex. ``tanh_l1``: long-short unit-L1.
    ``raw``: identity (smoke / bakeoff only).
    ``sparse_tilt`` / ``sparse_tilt_tsallis``: Euclidean sparsemax around
    baseline-relative logits (same geometry; Tsallis entropy gated in PPO).
    ``entmax_15``: Tsallis-1.5 entmax around baseline-relative logits.
    ``dirichlet_tilt`` / ``dirichlet_mean`` / ``dirichlet_entropy``: ``raw`` is
    a simplex sample ``u``; apply multiplicative tilt around ``w_base``.
    """
    key = str(head or "softmax").lower()
    if key == "softmax":
        temp = max(float(temperature), 1e-6)
        gain = max(float(tilt_gain), 1e-6)
        return F.softmax(raw * gain / temp, dim=-1)
    if key == "tanh_l1":
        z = torch.tanh(raw)
        denom = z.abs().sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return z / denom
    if key == "raw":
        return raw
    if key in ("sparse_tilt", "sparse_tilt_tsallis"):
        from mascotrl.policy.sparsemax import sparsemax

        temp = max(float(temperature), 1e-6)
        gain = max(float(tilt_gain), 1e-6)
        logits = raw * gain / temp
        if w_base is not None:
            base = w_base
            if base.ndim < logits.ndim:
                base = base.unsqueeze(0).expand_as(logits)
            base_logits = torch.log(base.clamp(min=1e-8))
            logits = base_logits + logits
        return sparsemax(logits, dim=-1)
    if key == "entmax_15":
        from mascotrl.policy.entmax import entmax

        temp = max(float(temperature), 1e-6)
        gain = max(float(tilt_gain), 1e-6)
        logits = raw * gain / temp
        if w_base is not None:
            base = w_base
            if base.ndim < logits.ndim:
                base = base.unsqueeze(0).expand_as(logits)
            base_logits = torch.log(base.clamp(min=1e-8))
            logits = base_logits + logits
        return entmax(logits, alpha=1.5, dim=-1)
    if key == "dirichlet_tilt":
        from mascotrl.policy.dirichlet_tilt import multiplicative_tilt

        return multiplicative_tilt(
            raw, w_base=w_base, mask=mask, kappa=float(tilt_gain)
        )
    if key in ("dirichlet_mean", "dirichlet_entropy"):
        # Deterministic / off-policy adapters: raw is logits → Dir mean → tilt.
        from mascotrl.policy.dirichlet_tilt import (
            concentrations_from_logits,
            multiplicative_tilt,
        )

        alpha = concentrations_from_logits(raw)
        u = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return multiplicative_tilt(
            u, w_base=w_base, mask=mask, kappa=float(tilt_gain)
        )
    raise ValueError(f"unknown weight_head={head!r}")


def _orthogonal_init(module: nn.Module, *, gain: float = 1.0) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class _ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: int = 64,
        *,
        actor_final_gain: float = 0.1,
    ):
        super().__init__()
        self.actor = _mlp(obs_dim, action_dim, hidden)
        self.critic = _mlp(obs_dim, 1, hidden)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        # Small final-layer gain for actor; critic unit gain.
        self.actor.apply(lambda m: _orthogonal_init(m, gain=actor_final_gain))
        self.critic.apply(lambda m: _orthogonal_init(m, gain=1.0))
        # Re-init final actor layer even smaller.
        last = self.actor[-1]
        if isinstance(last, nn.Linear):
            nn.init.orthogonal_(last.weight, gain=actor_final_gain)
            nn.init.zeros_(last.bias)

    def mean(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


class _AssetTemporalActorCritic(nn.Module):
    """C2: architecture-axis body for research PPO (delegates to bodies.py)."""

    def __init__(
        self,
        num_assets: int,
        d_model: int,
        action_dim: int,
        *,
        seq_len: int = 1,
        d_state: int = 16,
        temporal_backend: str = "gru",
        share_temporal_encoder: bool = True,
        use_surface_image_encoder: bool = False,
        image_channels: int = 0,
        surface_image_embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.body = AssetTemporalPolicyBody(
            num_assets=num_assets,
            d_model=d_model,
            action_dim=action_dim,
            seq_len=seq_len,
            d_state=d_state,
            temporal_backend=temporal_backend,
            share_temporal_encoder=share_temporal_encoder,
            use_surface_image_encoder=use_surface_image_encoder,
            image_channels=image_channels,
            surface_image_embed_dim=surface_image_embed_dim,
            with_critic=True,
        )
        self.log_std = nn.Parameter(torch.full((int(action_dim),), -0.5))
        # Compatibility aliases for tests / diagnostics that introspect the
        # pre-bodies layout on the actor-critic wrapper.
        self.extractor = self.body.extractor
        self.use_surface_image_encoder = self.body.use_surface_image_encoder
        self.image_encoder = self.body.image_encoder
        self.image_channels = self.body.image_channels
        self.num_assets = self.body.num_assets
        self.d_model = self.body.d_model
        self.seq_len = self.body.seq_len
        self.base_d_model = self.body.base_d_model
        self.actor_head = self.body.actor_head
        self.critic_head = self.body.critic_head

    def mean(self, obs: torch.Tensor) -> torch.Tensor:
        return self.body.mean(obs)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.body.value(obs)


class _QNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.net = _mlp(obs_dim + action_dim, 1, hidden)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


class RunningMeanStd:
    """Welford running observation normalizer (frozen at eval when desired)."""

    def __init__(self, dim: int, *, epsilon: float = 1e-8) -> None:
        self.mean = torch.zeros(dim)
        self.var = torch.ones(dim)
        self.count = epsilon
        self.epsilon = float(epsilon)
        self.frozen = False

    def update(self, x: torch.Tensor) -> None:
        if self.frozen:
            return
        x = x.detach().reshape(-1, self.mean.numel())
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = float(x.shape[0])
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / torch.sqrt(
            self.var.to(x.device) + self.epsilon
        )


class _BaseAgent:
    """Common act / train_epoch surface for bakeoff adapters."""

    name: str = "base"
    backend: str = "custom"

    def act(self, obs: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        raise NotImplementedError

    def train_epoch(
        self,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, float]:
        raise NotImplementedError


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation; returns (advantages, returns)."""
    t_len = int(rewards.shape[0])
    adv = torch.zeros_like(rewards)
    last = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(t_len)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * mask - values[t]
        last = delta + gamma * gae_lambda * mask * last
        adv[t] = last
    returns = adv + values
    return adv, returns


def _explained_variance(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    var_y = float(torch.var(y_true, unbiased=False))
    if var_y < 1e-12:
        return float("nan")
    return float(1.0 - torch.var(y_true - y_pred, unbiased=False) / var_y)

# Public aliases for internal names used across agent modules.
mlp = _mlp
actor_body = _actor_body
apply_weight_head = _apply_weight_head
orthogonal_init = _orthogonal_init
ActorCritic = _ActorCritic
AssetTemporalActorCritic = _AssetTemporalActorCritic
QNet = _QNet
BaseAgent = _BaseAgent
explained_variance = _explained_variance
WEIGHT_HEADS = _WEIGHT_HEADS
