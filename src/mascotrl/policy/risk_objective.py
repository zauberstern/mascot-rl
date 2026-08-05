"""Rockafellar-Uryasev CVaR, Buehler entropic, and spectrum risk objectives."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Legacy short names + spectrum ids.
ALLOWED_MODES = frozenset(
    {
        "none",
        "cvar",
        "cvar_ru",
        "entropic",
        "entropic_oce",
        "smse",
        "rsqp",
        "mean_std_cao",
        "meanvar_kolm",
        "mtm_pnl",
        "differential_sharpe",
        "mikkila_asym",
    }
)

_MODE_ALIASES = {
    "cvar_ru": "cvar",
    "entropic_oce": "entropic",
    "mtm_pnl": "none",
    # differential_sharpe / mikkila_asym: keep mode identity; loss returns 0
    # (dense online path handled outside this module).
}


class RiskObjective(nn.Module):
    """Optional risk / mean-dispersion objective over episode returns G.

    Modes
    -----
    none / mtm_pnl
        Zero loss here (primary handled elsewhere).
    differential_sharpe / mikkila_asym
        Zero loss here; dense online reward path (not critic overlay).
    cvar / cvar_ru
        Rockafellar-Uryasev CVaR of losses L = -G.
    entropic / entropic_oce
        Buehler entropic risk.
    smse
        Semi-mean-squared error on positive losses (Francois).
    rsqp
        Root semi-quadratic penalty (Neagu).
    mean_std_cao
        E[G] penalty inverted: -(mean) + c * std (minimized as loss).
    meanvar_kolm
        -mean + (kappa/2) * var.
    """

    def __init__(
        self,
        mode: str = "none",
        alpha: float = 0.95,
        lam: float = 1.0,
        coef: float = 0.0,
        zeta_lr: float = 3e-3,
        cao_c: float = 1.5,
        kappa: float = 1.0,
        objective_primary: bool = False,
    ):
        super().__init__()
        raw = str(mode)
        if raw not in ALLOWED_MODES:
            raise ValueError(f"unknown mode={mode!r}; expected one of {sorted(ALLOWED_MODES)}")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if lam <= 0.0:
            raise ValueError(f"lam must be positive, got {lam}")

        self.raw_mode = raw
        self.mode = _MODE_ALIASES.get(raw, raw)
        self.alpha = float(alpha)
        self.lam = float(lam)
        self.coef = float(coef)
        self.zeta_lr = float(zeta_lr)
        self.cao_c = float(cao_c)
        self.kappa = float(kappa)
        self.objective_primary = bool(objective_primary)
        # D.8: optional policy_mode risk-aversion overlay (applied lazily).
        self.policy_mode = "shared"
        self._term_spread_z: float | None = None

        if self.mode == "cvar":
            self.zeta = nn.Parameter(torch.zeros(()))
            self._zeta_opt = torch.optim.SGD([self.zeta], lr=self.zeta_lr)
        else:
            self.register_parameter("zeta", None)
            self._zeta_opt = None

    def apply_policy_mode(
        self,
        policy_mode: str,
        *,
        term_spread_z: float | None = None,
    ) -> None:
        """Scale cao_c / kappa by the mandate risk-aversion multiplier."""
        from mascotrl.spectrum.policy_mode import apply_risk_aversion, resolve_policy_mode

        self.policy_mode = resolve_policy_mode({"policy_mode": policy_mode})
        self._term_spread_z = term_spread_z
        # Re-base from constructor values stored on first call.
        if not hasattr(self, "_base_cao_c"):
            self._base_cao_c = float(self.cao_c)
            self._base_kappa = float(self.kappa)
        self.cao_c = apply_risk_aversion(
            self._base_cao_c, self.policy_mode, term_spread_z=term_spread_z
        )
        self.kappa = apply_risk_aversion(
            self._base_kappa, self.policy_mode, term_spread_z=term_spread_z
        )

    @property
    def zeta_value(self) -> float | None:
        """Current ζ for logging; None when mode is not cvar."""
        if self.zeta is None:
            return None
        return float(self.zeta.detach())

    def loss(self, episode_returns: torch.Tensor) -> torch.Tensor:
        """Scalar risk / dispersion loss on episode returns G (minimize)."""
        g = episode_returns.reshape(-1)
        # Primary objectives always apply; overlay modes respect coef==0.
        if self.mode == "none" or self.mode in ("differential_sharpe", "mikkila_asym"):
            return 0.0 * g.sum()
        if (
            not self.objective_primary
            and self.coef == 0.0
            and self.mode in ("cvar", "entropic")
        ):
            return 0.0 * g.sum()

        coef = self.coef if self.coef != 0.0 or not self.objective_primary else 1.0

        if self.mode == "cvar":
            excess = F.relu(-g - self.zeta)
            cvar = self.zeta + excess.mean() / (1.0 - self.alpha)
            return coef * cvar

        if self.mode == "entropic":
            n = g.numel()
            log_mean_exp = torch.logsumexp(-self.lam * g, dim=0) - math.log(n)
            return coef * (log_mean_exp / self.lam)

        if self.mode == "smse":
            # Semi-MSE on losses L=-G (positive part of -G).
            loss_pos = F.relu(-g)
            return coef * (loss_pos.pow(2).mean())

        if self.mode == "rsqp":
            loss_pos = F.relu(-g)
            return coef * torch.sqrt(loss_pos.pow(2).mean() + 1e-12)

        if self.mode == "mean_std_cao":
            # Minimize -(mean) + c*std  <=> maximize mean - c*std
            mu = g.mean()
            std = g.std(unbiased=False)
            return coef * (-mu + self.cao_c * std)

        if self.mode == "meanvar_kolm":
            mu = g.mean()
            var = g.var(unbiased=False)
            return coef * (-mu + 0.5 * self.kappa * var)

        return 0.0 * g.sum()

    def step_zeta(self) -> None:
        """Optional faster-timescale SGD step on ζ (call after backward)."""
        if self._zeta_opt is None:
            return
        self._zeta_opt.step()
        self._zeta_opt.zero_grad()
