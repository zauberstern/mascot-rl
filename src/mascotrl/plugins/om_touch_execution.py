"""OM-touch / half-spread execution cost measurement (Buehler-compatible).

Hard CMDP projection remains the constraint layer. This module measures
buy@offer / sell@bid (or mid±half-spread) costs for reporting / OOS / shadow.
It must NOT replace τ/δ projection with soft Almgren penalties.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


# Fraction of the quoted half-spread actually paid. Muravyev and Pearson
# (2020, RFS 33(11)) find conventional effective spreads overstate the cost of
# taking liquidity: the average effective spread is about one quarter smaller
# than conventional measures, and under 40% for traders who time executions.
# The full quoted half-spread (1.0) is therefore an upper bound, not a estimate.
SPREAD_MULTIPLIER_LADDER: tuple[float, ...] = (0.0, 0.25, 0.50, 1.0)


@dataclass
class OMTouchCost:
    """
    Per-step touch cost from half-spreads.

    ``half_spreads``: (B, K) or (K,) absolute price half-spreads in the same
    units as mid (the lake column is already a half-spread; do not halve again).

    ``capital_base``: (B, K) or (K,) dollars of invested capital per unit of the
    traded package — for delta-hedged options this is (Δ·S − C). Weights are
    fractions of capital, so a dollar half-spread must be divided by the capital
    base to land in return units::

        cost_i = |Δw_i| * half_spread_i / capital_base_i

    Omitting ``capital_base`` reverts to the legacy dollar-P&L convention, which
    is only unit-correct when weights are contract counts.

    ``spread_multiplier`` scales the quoted half-spread for the cost ladder.
    """

    enabled: bool = False
    fee_bps: float = 0.0
    spread_multiplier: float = 1.0
    model_name: str = "om_touch"

    def compute(
        self,
        w_exec: torch.Tensor,
        w_prev: torch.Tensor,
        half_spreads: torch.Tensor | None,
        capital_base: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros(w_exec.shape[0], device=w_exec.device, dtype=w_exec.dtype)
        dw = (w_exec - w_prev).abs()
        if half_spreads is None:
            touch = torch.zeros(w_exec.shape[0], device=w_exec.device, dtype=w_exec.dtype)
        else:
            hs = half_spreads
            if hs.dim() == 1:
                hs = hs.unsqueeze(0).expand_as(dw)
            hs = hs.to(device=dw.device, dtype=dw.dtype)
            per_name = dw * hs * float(self.spread_multiplier)
            if capital_base is not None:
                cb = capital_base
                if cb.dim() == 1:
                    cb = cb.unsqueeze(0).expand_as(dw)
                cb = cb.to(device=dw.device, dtype=dw.dtype)
                # Degenerate or non-positive capital bases contribute nothing
                # rather than exploding the cost.
                safe = cb.abs() > 1e-8
                per_name = torch.where(
                    safe, per_name / torch.where(safe, cb.abs(), torch.ones_like(cb)),
                    torch.zeros_like(per_name),
                )
            touch = per_name.sum(dim=-1)
        fee = (self.fee_bps / 1e4) * dw.sum(dim=-1)
        return touch + fee


def apply_om_touch_to_pnl(
    pnl: float,
    w: torch.Tensor,
    w_prev: torch.Tensor,
    *,
    half_spreads: torch.Tensor | None,
    fee_bps: float = 0.0,
    enabled: bool = True,
    spread_multiplier: float = 1.0,
    capital_base: torch.Tensor | None = None,
) -> tuple[float, float]:
    """Return (net_pnl, touch_drag)."""
    if not enabled:
        return float(pnl), 0.0
    cost = OMTouchCost(
        enabled=True, fee_bps=fee_bps, spread_multiplier=spread_multiplier
    )
    drag = float(
        cost.compute(w, w_prev, half_spreads, capital_base).reshape(-1)[0].item()
    )
    return float(pnl) - drag, drag
