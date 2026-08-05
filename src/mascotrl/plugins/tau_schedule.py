"""Exogenous turnover-limit schedules (never actor/critic-chosen)."""
from __future__ import annotations

from typing import Protocol

import torch


class TauSchedule(Protocol):
    def __call__(
        self,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        macro: torch.Tensor | None = None,
        atm_vol: float | None = None,
    ) -> torch.Tensor: ...


class FixedTau:
    """Status-quo: constant τ = turnover_limit."""

    def __init__(self, tau0: float = 0.15):
        self.tau0 = float(tau0)

    def __call__(
        self,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        macro: torch.Tensor | None = None,
        atm_vol: float | None = None,
    ) -> torch.Tensor:
        return torch.full((batch,), self.tau0, device=device, dtype=dtype)


class MacroScheduleTau:
    """
    Exogenous regime schedule:

        τ_t = clip(τ₀ · (1 + α · (z_VIX − z_ref)), τ_min, τ_max)

    ``macro`` is expected shaped (B, macro_dim) or (macro_dim,) with VIX z at
    ``vix_macro_index`` (default 0 — first column after load_macro_tensor z-score).
    """

    def __init__(
        self,
        tau0: float = 0.15,
        tau_min: float = 0.05,
        tau_max: float = 0.40,
        vix_z_ref: float = 0.0,
        vix_z_scale: float = 0.25,
        vix_macro_index: int = 0,
    ):
        self.tau0 = float(tau0)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.vix_z_ref = float(vix_z_ref)
        self.vix_z_scale = float(vix_z_scale)
        self.vix_macro_index = int(vix_macro_index)

    def __call__(
        self,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        macro: torch.Tensor | None = None,
        atm_vol: float | None = None,
    ) -> torch.Tensor:
        if macro is None:
            return torch.full((batch,), self.tau0, device=device, dtype=dtype)
        m = macro
        if m.dim() == 1:
            m = m.unsqueeze(0)
        if m.shape[0] == 1 and batch > 1:
            m = m.expand(batch, -1)
        idx = min(self.vix_macro_index, m.shape[-1] - 1)
        z = m[:, idx].to(device=device, dtype=dtype)
        tau = self.tau0 * (1.0 + self.vix_z_scale * (z - self.vix_z_ref))
        return tau.clamp(self.tau_min, self.tau_max)
