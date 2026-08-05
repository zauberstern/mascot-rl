"""HAPPO-style sequential factorized PPO with centralized critic + elastic QP.

Stochastic policy: diagonal Gaussian over raw weight increments Δw. Log-probabilities
and PPO ratios are defined on this raw Δw. The elastic QP maps (w_prev+Δw) to a
feasible executed weight w_exec deterministically; that map is outside the
score-function policy density.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn

from mascotrl.policy.convex_projection import ConvexProjectionLayer
from mascotrl.plugins.hypernet_actors import DeepSetsCritic, HypernetActors
from mascotrl.plugins.tau_schedule import FixedTau


def layer_init(layer: nn.Linear, std: float = math.sqrt(2.0)) -> nn.Linear:
    """Orthogonal init (Huang et al. 37 PPO details). Biases → 0."""
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def _mlp_actor(enriched_dim: int) -> nn.Sequential:
    """Two-layer actor: hidden gain √2, mean head gain 0.01."""
    return nn.Sequential(
        layer_init(nn.Linear(enriched_dim, 64), std=math.sqrt(2.0)),
        nn.GELU(),
        layer_init(nn.Linear(64, 1), std=0.01),
    )


def _mlp_critic(in_dim: int) -> nn.Sequential:
    """Critic: hidden √2, value head gain 1.0."""
    return nn.Sequential(
        layer_init(nn.Linear(in_dim, 128), std=math.sqrt(2.0)),
        nn.GELU(),
        layer_init(nn.Linear(128, 1), std=1.0),
    )


# @lat: [[core#HAPPO]]
class HAPPOEngine(nn.Module):
    """K independent actors + centralized critic + elastic QP projection."""

    def __init__(
        self,
        num_assets: int,
        enriched_dim: int,
        macro_dim: int,
        turnover_limit: float = 0.15,
        *,
        use_projection: bool = True,
        max_name_abs_weight: float = 5.0,
        actor_backend: str = "modulelist",
        critic_backend: str = "flatten",
        hypernet_cfg: dict[str, Any] | None = None,
        initial_log_std: float = -2.0,
        enable_cost_critic: bool = False,
        actor_portfolio_state: bool = False,
    ):
        super().__init__()
        self.num_assets = num_assets
        self.enriched_dim = int(enriched_dim)
        self.macro_dim = int(macro_dim)
        self.use_projection = bool(use_projection)
        self.max_name_abs_weight = float(max_name_abs_weight)
        self.actor_backend = str(actor_backend)
        self.critic_backend = str(critic_backend)
        self.initial_log_std = float(initial_log_std)
        self.enable_cost_critic = bool(enable_cost_critic)
        self.actor_portfolio_state = bool(actor_portfolio_state)
        hn = hypernet_cfg or {}
        # Persist for clone_happo_engine / WFO (custom embed_dim/hidden).
        self.hypernet_cfg = dict(hn)
        actor_in = int(enriched_dim) + (1 if self.actor_portfolio_state else 0)

        # shared / shared_mappo: one MLP for all assets (eq_alloc Phase N1).
        # modulelist remains the status-quo / paper-protocol ablation.
        if self.actor_backend == "hypernet":
            self.hypernet_actors = HypernetActors(
                num_assets,
                actor_in,
                embed_dim=int(hn.get("embed_dim", 16)),
                hidden=int(hn.get("hidden", 64)),
                condition_on=list(hn.get("condition_on") or []),
            )
            # Empty shim so clone_happo_engine / len(actors) paths don't crash.
            self.actors = nn.ModuleList()
        elif self.actor_backend in ("shared", "shared_mappo"):
            self.hypernet_actors = None
            self.actors = nn.ModuleList([_mlp_actor(actor_in)])
        else:
            # Status-quo checkpoint keys: actors.{i}.{0,2}.*
            self.actors = nn.ModuleList(
                [_mlp_actor(actor_in) for _ in range(num_assets)]
            )
            self.hypernet_actors = None

        if self.critic_backend == "deepsets":
            self.critic = DeepSetsCritic(enriched_dim, macro_dim)
            self._deepsets_critic = True
        else:
            # Status-quo checkpoint keys: critic.0.*, critic.2.*
            self.critic = _mlp_critic(num_assets * enriched_dim + macro_dim)
            self._deepsets_critic = False

        if self.enable_cost_critic:
            if self.critic_backend == "deepsets":
                self.cost_critic = DeepSetsCritic(enriched_dim, macro_dim)
            else:
                self.cost_critic = _mlp_critic(num_assets * enriched_dim + macro_dim)
        else:
            self.cost_critic = None

        # Skip SCS/CvxpyLayer construct when projection is off or will be
        # replaced (K>50 ADMM via build_happo_engine). Avoids multi-minute DPP
        # compile that stalls narrative HAPPO at K=100.
        self.convex_projection = None
        if self.use_projection:
            self.convex_projection = ConvexProjectionLayer(
                num_assets,
                turnover_limit=turnover_limit,
                max_name_abs_weight=self.max_name_abs_weight,
            )
        self._projection_backend = "cvxpy" if self.use_projection else "none"
        # Per-agent log_std so modulelist Adam steps cannot dual-update all σ.
        self._log_std = nn.ParameterList(
            [
                nn.Parameter(torch.tensor(float(self.initial_log_std)))
                for _ in range(num_assets)
            ]
        )
        self.tau_schedule = FixedTau(tau0=float(turnover_limit))

    @property
    def log_std(self) -> torch.Tensor:
        """Stacked (K,) log-std for Gaussian sampling / entropy / stats."""
        return torch.stack([p for p in self._log_std])

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        """Accept legacy flat ``log_std`` checkpoints and ParameterList keys."""
        sd = dict(state_dict)
        if "log_std" in sd and "log_std.0" not in sd and "_log_std.0" not in sd:
            flat = sd.pop("log_std")
            if torch.is_tensor(flat) and flat.numel() >= self.num_assets:
                for i in range(self.num_assets):
                    sd[f"_log_std.{i}"] = flat.reshape(-1)[i].detach().clone()
        for i in range(self.num_assets):
            alt = f"log_std.{i}"
            if alt in sd and f"_log_std.{i}" not in sd:
                sd[f"_log_std.{i}"] = sd.pop(alt)
        return super().load_state_dict(sd, strict=strict)

    def _resolve_tau(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        macro: torch.Tensor | None = None,
        turnover_limit: torch.Tensor | float | None = None,
    ) -> torch.Tensor | float | None:
        if turnover_limit is not None:
            return turnover_limit
        return self.tau_schedule(
            batch=batch, device=device, dtype=dtype, macro=macro
        )

    def _actor_input(
        self, enriched_states: torch.Tensor, w_prev: torch.Tensor | None
    ) -> torch.Tensor:
        """Optionally concat per-slot ``w_prev`` (``actor_portfolio_state``)."""
        if not self.actor_portfolio_state:
            return enriched_states
        if w_prev is None:
            raise ValueError("actor_portfolio_state=True requires w_prev")
        w = w_prev
        if w.dim() == 1:
            w = w.unsqueeze(0)
        if w.shape[0] != enriched_states.shape[0] or w.shape[-1] != enriched_states.shape[1]:
            raise ValueError(
                f"w_prev shape {tuple(w.shape)} incompatible with "
                f"enriched {tuple(enriched_states.shape)}"
            )
        return torch.cat([enriched_states, w.unsqueeze(-1)], dim=-1)

    def _actor_means(
        self,
        enriched_states: torch.Tensor,
        w_prev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        states = self._actor_input(enriched_states, w_prev)
        if self.actor_backend == "hypernet" and self.hypernet_actors is not None:
            return self.hypernet_actors._actor_means(states)
        if self.actor_backend in ("shared", "shared_mappo"):
            actor = self.actors[0]
            b, k, d = states.shape
            flat = states.reshape(b * k, d)
            return actor(flat).view(b, k)
        parts = [
            actor(states[:, i, :]) for i, actor in enumerate(self.actors)
        ]
        return torch.cat(parts, dim=-1)

    def _value(
        self, enriched_states: torch.Tensor, macro_features: torch.Tensor
    ) -> torch.Tensor:
        if getattr(self, "_deepsets_critic", False):
            return self.critic(enriched_states, macro_features)
        global_state = torch.cat(
            [enriched_states.reshape(enriched_states.shape[0], -1), macro_features],
            dim=-1,
        )
        return self.critic(global_state).squeeze(-1)

    def _cost_value(
        self, enriched_states: torch.Tensor, macro_features: torch.Tensor
    ) -> torch.Tensor:
        if self.cost_critic is None:
            return torch.zeros(
                enriched_states.shape[0],
                device=enriched_states.device,
                dtype=enriched_states.dtype,
            )
        if getattr(self, "_deepsets_critic", False):
            return self.cost_critic(enriched_states, macro_features)
        global_state = torch.cat(
            [enriched_states.reshape(enriched_states.shape[0], -1), macro_features],
            dim=-1,
        )
        return self.cost_critic(global_state).squeeze(-1)

    def _project(
        self,
        w_target: torch.Tensor,
        w_prev: torch.Tensor,
        deltas: torch.Tensor,
        vol_scale: torch.Tensor | float | None,
        *,
        macro: torch.Tensor | None = None,
        turnover_limit: torch.Tensor | float | None = None,
        return_slacks: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_projection:
            if return_slacks:
                z = torch.zeros(w_target.shape[0], device=w_target.device, dtype=w_target.dtype)
                return w_target, z, z
            return w_target
        if self.convex_projection is None:
            raise RuntimeError("projection_layer_missing")
        tau = self._resolve_tau(
            w_target.shape[0],
            w_target.device,
            w_target.dtype,
            macro=macro,
            turnover_limit=turnover_limit,
        )
        return self.convex_projection(
            w_target,
            w_prev,
            deltas,
            vol_scale=vol_scale,
            turnover_limit=tau,
            return_slacks=return_slacks,
        )

    def forward(
        self,
        enriched_states: torch.Tensor,
        macro_features: torch.Tensor,
        w_prev: torch.Tensor,
        deltas: torch.Tensor,
        vol_scale: torch.Tensor | float | None = None,
        turnover_limit: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta_w = self._actor_means(enriched_states, w_prev)
        w_target = w_prev + delta_w
        w_exec = self._project(
            w_target,
            w_prev,
            deltas,
            vol_scale,
            macro=macro_features,
            turnover_limit=turnover_limit,
        )
        return w_exec, self._value(enriched_states, macro_features)

    def act_deterministic(
        self,
        enriched_states: torch.Tensor,
        macro_features: torch.Tensor,
        w_prev: torch.Tensor,
        deltas: torch.Tensor,
        vol_scale: torch.Tensor | float | None = None,
        turnover_limit: torch.Tensor | float | None = None,
        return_slacks: bool = False,
    ) -> tuple:
        delta_w = self._actor_means(enriched_states, w_prev)
        proj_out = self._project(
            w_prev + delta_w,
            w_prev,
            deltas,
            vol_scale,
            macro=macro_features,
            turnover_limit=turnover_limit,
            return_slacks=return_slacks,
        )
        value = self._value(enriched_states, macro_features)
        if return_slacks:
            w_exec, s_delta, s_turn = proj_out
            return w_exec, value, delta_w, s_delta, s_turn
        return proj_out, value, delta_w

    def act_stochastic(
        self,
        enriched_states: torch.Tensor,
        macro_features: torch.Tensor,
        w_prev: torch.Tensor,
        deltas: torch.Tensor,
        vol_scale: torch.Tensor | float | None = None,
        turnover_limit: torch.Tensor | float | None = None,
        return_slacks: bool = False,
    ) -> tuple:
        mean = self._actor_means(enriched_states, w_prev)
        std = self.log_std.exp().unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        delta_w_raw = dist.rsample()
        log_probs = dist.log_prob(delta_w_raw)
        w_target = w_prev + delta_w_raw
        proj_out = self._project(
            w_target,
            w_prev,
            deltas,
            vol_scale,
            macro=macro_features,
            turnover_limit=turnover_limit,
            return_slacks=return_slacks,
        )
        if return_slacks:
            w_exec, s_delta, s_turn = proj_out
        else:
            w_exec = proj_out
            s_delta = s_turn = None
        value = self._value(enriched_states, macro_features)
        if return_slacks:
            return w_exec, log_probs, value, delta_w_raw, s_delta, s_turn
        return w_exec, log_probs, value, delta_w_raw

    def evaluate_raw_log_probs(
        self,
        enriched_states: torch.Tensor,
        w_raw: torch.Tensor,
        w_prev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mean = self._actor_means(enriched_states, w_prev)
        std = self.log_std.exp().unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(w_raw)

    @staticmethod
    def gaussian_entropy(log_std_k: torch.Tensor) -> torch.Tensor:
        return 0.5 * (math.log(2.0 * math.pi * math.e) + 2.0 * log_std_k)
