"""OmniSafe-backed CPPO adapter (vendored PID/naive Lagrange, no Safety-Gymnasium).

Maps MascotRL's CVaR-constraint signal into a scalar episode cost for OmniSafe's
``PIDLagrangian`` / ``Lagrange``, then mixes cost into the PPO advantage like
OmniSafe ``CPPOPID``:

    L = 1/(1+λ) * (A_r - λ A_c)

Cost shaping (CVaR tail): ``cost_t = relu(-r_t - zeta)`` with zeta = empirical
VaR of batch returns. Does **not** soft-fee overnight ``R_t``; the env reward
stream stays the estimand, and cost is a separate constraint channel.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from mascotrl.logging_utils import get_logger
from mascotrl.policy.single_agent import PPOAgent, compute_gae
from mascotrl.policy.vendor.omnisafe import Lagrange, PIDLagrangian

log = get_logger("mascotrl.policy.omnisafe")


class CostShaper:
    """Map rewards to a non-negative cost for the constrained dual."""

    def __init__(self, *, cvar_alpha: float = 0.95):
        self.cvar_alpha = float(cvar_alpha)
        self.zeta = 0.0

    def episode_cost(self, rewards: torch.Tensor, dones: torch.Tensor) -> float:
        """Mean per-step tail cost over the batch (proxy for EpCost)."""
        r = rewards.detach().reshape(-1).float()
        losses = -r
        k = max(1, int(round((1.0 - self.cvar_alpha) * max(int(r.numel()), 1))))
        top = torch.topk(losses, k=k, largest=True).values
        self.zeta = float(top.min().item()) if top.numel() else 0.0
        cost = F.relu(losses - self.zeta)
        return float(cost.mean().item())

    def cost_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
        *,
        gamma: float,
        gae_lambda: float,
    ) -> torch.Tensor:
        r = rewards.detach().reshape(-1).float()
        costs = F.relu(-r - self.zeta)
        # Cost critic bootstrap with zeros (no separate cost value net).
        zeros = torch.zeros_like(values)
        adv_c, _ = compute_gae(
            costs,
            zeros,
            zeros,
            dones,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        return adv_c


class OmniSafeCPPOAgent(PPOAgent):
    """PPO + vendored OmniSafe CPPOPID / PPOLag dual.

    ``omnisafe_algo``: ``cppo_pid`` (default) or ``ppo_lag``.
    """

    name = "cppo_omnisafe"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        cvar_alpha: float = 0.95,
        cvar_k_ratio: float = 0.2,
        nu_lr: float = 0.01,
        nu_delay: float = 0.2,
        omnisafe_algo: str = "cppo_pid",
        cost_limit: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(obs_dim, action_dim, **kwargs)
        self.normalize_adv = False  # mixed adv must not be re-normalized away
        self.cvar_alpha = float(cvar_alpha)
        self.cvar_k_ratio = float(cvar_k_ratio)
        self.nu_lr = float(nu_lr)
        self.nu_delay = float(nu_delay)
        self.omnisafe_algo = str(omnisafe_algo or "cppo_pid").lower().strip()
        self.cost_shaper = CostShaper(cvar_alpha=self.cvar_alpha)
        self._adv_c: torch.Tensor | None = None
        if self.omnisafe_algo in {"cppo_pid", "cppopid", "pid"}:
            self._lagrange: PIDLagrangian | Lagrange = PIDLagrangian(
                pid_kp=float(nu_lr),
                pid_ki=float(nu_lr) * 0.1,
                pid_kd=float(nu_delay),
                pid_d_delay=10,
                pid_delta_p_ema_alpha=0.95,
                pid_delta_d_ema_alpha=0.95,
                sum_norm=True,
                diff_norm=False,
                penalty_max=100,
                lagrangian_multiplier_init=0.0,
                cost_limit=float(cost_limit),
            )
            self._dual_kind = "pid"
        else:
            self._lagrange = Lagrange(
                cost_limit=float(cost_limit),
                lagrangian_multiplier_init=0.0,
                lambda_lr=float(nu_lr),
                lambda_optimizer="Adam",
                lagrangian_upper_bound=100.0,
            )
            self._dual_kind = "naive"

    def _constraint_penalty(self, returns_mb: torch.Tensor) -> torch.Tensor:
        # Dual enters via advantage mixing; keep hook zero to avoid double count.
        return torch.zeros((), device=returns_mb.device, dtype=returns_mb.dtype)

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
        x = self._prep_obs(obs, update_rms=False)
        with torch.no_grad():
            values = self.net.value(x)
            next_x = self._prep_obs(next_obs, update_rms=False)
            next_values = self.net.value(next_x)
            adv_r, returns = compute_gae(
                rewards,
                values.flatten(),
                next_values.flatten(),
                dones,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )
        ep_cost = self.cost_shaper.episode_cost(rewards, dones)
        if self._dual_kind == "pid":
            assert isinstance(self._lagrange, PIDLagrangian)
            self._lagrange.pid_update(ep_cost)
            lam = float(self._lagrange.lagrangian_multiplier)
        else:
            assert isinstance(self._lagrange, Lagrange)
            self._lagrange.update_lagrange_multiplier(ep_cost)
            lam = float(self._lagrange.lagrangian_multiplier.detach().item())

        adv_c = self.cost_shaper.cost_advantages(
            rewards,
            values.flatten(),
            next_values.flatten(),
            dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        # CPPOPID surrogate: (A_r - λ A_c) / (1 + λ)
        mixed = (adv_r - lam * adv_c) / (1.0 + lam)
        # Feed mixed advantages by scaling sample_weight relative to GAE.
        # PPOAgent multiplies advantages by sample_weight then z-norms (RC5);
        # we override via sample_weight = mixed / (adv_r + eps).
        sw = mixed / (adv_r.detach() + 1e-8)
        if sample_weight is not None:
            sw = sw * sample_weight.detach().reshape(-1).to(sw.dtype)

        stats = super().train_epoch(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
            old_logprobs=old_logprobs,
            n_epochs=n_epochs,
            n_minibatches=n_minibatches,
            sample_weight=sw,
            policy_step_mask=policy_step_mask,
            scr_mix=scr_mix,
            scr_beta=scr_beta,
            scr_y_cf=scr_y_cf,
        )
        stats["omnisafe_lambda"] = float(lam)
        stats["omnisafe_ep_cost"] = float(ep_cost)
        stats["omnisafe_algo"] = 1.0 if self._dual_kind == "pid" else 0.0
        stats["cvar_zeta"] = float(self.cost_shaper.zeta)
        return stats
