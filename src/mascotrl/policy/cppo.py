"""CVaR Proximal Policy Optimization (CPPO, Ying et al. IJCAI 2022).

Trajectory-CVaR-constrained PPO via Lagrangian relaxation with dual ``nu`` and
VaR auxiliary ``eta``. Distinct from ``cvar_ru`` (Rockafellar-Uryasev soft objective).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from mascotrl.policy.single_agent import PPOAgent, compute_gae


class CPPOAgent(PPOAgent):
    """PPO with hard trajectory CVaR constraint (CPPO).

    Dual ascent on ``nu`` / ``eta`` runs before the PPO update; the Lagrangian
    term ``nu * relu(CVaR - beta)`` enters the actor-critic loss via
    :meth:`_constraint_penalty` so the constraint shapes policy gradients.
    """

    name = "cppo"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        cvar_alpha: float = 0.95,
        cvar_k_ratio: float = 0.2,
        nu_lr: float = 0.01,
        nu_delay: float = 0.2,
        **kwargs: Any,
    ):
        super().__init__(obs_dim, action_dim, **kwargs)
        self.cvar_alpha = float(cvar_alpha)
        self.cvar_k_ratio = float(cvar_k_ratio)
        self.nu_lr = float(nu_lr)
        self.nu_delay = float(nu_delay)
        self.eta = torch.zeros((), dtype=torch.float32)
        self.nu = torch.tensor(10.0)
        self._returns_buf: list[float] = []
        self._pending_violation = 0.0

    def _trajectory_cvar(self, returns: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rockafellar-Uryasev CVaR on losses L = -G."""
        g = returns.reshape(-1)
        losses = -g
        z = self.eta.to(device=g.device, dtype=g.dtype)
        tail = F.relu(losses - z)
        cvar = z + tail.mean() / max(1e-8, 1.0 - self.cvar_alpha)
        return cvar, losses

    def _update_beta(self, returns: torch.Tensor) -> float:
        g = returns.detach().reshape(-1)
        n = max(int(g.numel()), 1)
        k = max(1, int(round(n * self.cvar_k_ratio)))
        worst = torch.topk(-g, k=k, largest=True).values
        return float((-worst).mean().item())

    def _constraint_penalty(self, returns_mb: torch.Tensor) -> torch.Tensor:
        """Lagrangian: nu * relu(CVaR(returns) - beta_proxy)."""
        cvar, _ = self._trajectory_cvar(returns_mb)
        # Use last batch-level beta proxy stored during train_epoch prep.
        beta = float(getattr(self, "_last_beta", 0.0) or 0.0)
        violation = F.relu(cvar - (-beta))
        return self.nu.to(device=returns_mb.device, dtype=returns_mb.dtype) * violation

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
        # Prefetch returns to update duals before the PPO loop so nu/eta are
        # current when _constraint_penalty is evaluated inside minibatches.
        x = self._prep_obs(obs, update_rms=False)
        with torch.no_grad():
            values = self.net.value(x)
            next_x = self._prep_obs(next_obs, update_rms=False)
            next_values = self.net.value(next_x)
            _, returns = compute_gae(
                rewards,
                values.flatten(),
                next_values.flatten(),
                dones,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )
        beta = self._update_beta(returns)
        self._last_beta = beta
        cvar, losses = self._trajectory_cvar(returns)
        violation = cvar - (-beta)
        with torch.no_grad():
            self.nu = torch.clamp(
                self.nu + self.nu_lr * float(violation.item()), min=0.0
            )
            grad_eta = 1.0 - 1.0 / max(1e-8, 1.0 - self.cvar_alpha)
            if losses.numel() > 0:
                indicator = (losses >= self.eta.to(losses.device)).float().mean()
                self.eta = self.eta + self.nu_delay * (grad_eta - indicator)
        self._pending_violation = float(violation.item())

        stats = super().train_epoch(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
            old_logprobs=old_logprobs,
            n_epochs=n_epochs,
            n_minibatches=n_minibatches,
            sample_weight=sample_weight,
            policy_step_mask=policy_step_mask,
            scr_mix=scr_mix,
            scr_beta=scr_beta,
            scr_y_cf=scr_y_cf,
        )
        stats["cvar_eta"] = float(self.eta.item())
        stats["cvar_nu"] = float(self.nu.item())
        stats["cvar_beta"] = float(beta)
        stats["trajectory_cvar"] = float(cvar.item())
        stats["cvar_violation"] = float(self._pending_violation)
        return stats
