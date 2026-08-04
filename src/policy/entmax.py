"""Entmax-alpha and Tsallis entropy (Peters, Niculae, Martins 2019).

alpha=1 recovers softmax; alpha=2 recovers sparsemax; alpha=1.5 is the
standard intermediate used by the RC6_HEADS dose-response panel.
"""
from __future__ import annotations

import torch
from torch import Tensor

from src.policy.sparsemax import sparsemax


def entmax(z: Tensor, alpha: float = 1.5, dim: int = -1) -> Tensor:
    """Project ``z`` onto the Tsallis alpha-entmax simplex along ``dim``.

    Solves ``sum_j max((alpha-1)*z_j - tau, 0)^(1/(alpha-1)) = 1`` for ``tau``
    by bisection, then returns the corresponding sparse probabilities.
    """
    a = float(alpha)
    if abs(a - 1.0) < 1e-8:
        return torch.softmax(z, dim=dim)
    if abs(a - 2.0) < 1e-8:
        return sparsemax(z, dim=dim)

    if a <= 1.0:
        raise ValueError(f"entmax requires alpha > 1 (got {alpha})")

    p_exp = 1.0 / (a - 1.0)
    # Bound tau so support can go from full to empty.
    z_max = z.max(dim=dim, keepdim=True).values
    tau_lo = (a - 1.0) * z_max - 1.0
    tau_hi = (a - 1.0) * z_max

    for _ in range(40):
        tau = 0.5 * (tau_lo + tau_hi)
        p = ((a - 1.0) * z - tau).clamp(min=0.0).pow(p_exp)
        s = p.sum(dim=dim, keepdim=True)
        # If sum > 1, tau is too low (more mass); raise tau_lo.
        too_low = s > 1.0
        tau_lo = torch.where(too_low, tau, tau_lo)
        tau_hi = torch.where(too_low, tau_hi, tau)

    tau = 0.5 * (tau_lo + tau_hi)
    p = ((a - 1.0) * z - tau).clamp(min=0.0).pow(p_exp)
    # Numerical cleanup: renormalize tiny drift from finite bisection.
    return p / p.sum(dim=dim, keepdim=True).clamp_min(1e-12)


def tsallis_entropy(p: Tensor, alpha: float = 2.0, dim: int = -1) -> Tensor:
    """Tsallis entropy ``(1 - sum p^alpha) / (alpha - 1)`` along ``dim``.

    Zeros contribute 0 (NaN-safe). Returns a tensor with ``dim`` reduced.
    """
    a = float(alpha)
    if abs(a - 1.0) < 1e-8:
        # Shannon limit: -sum p log p
        p_safe = p.clamp(min=0.0)
        logp = torch.where(p_safe > 0, p_safe.log(), torch.zeros_like(p_safe))
        return -(p_safe * logp).sum(dim=dim)
    p_safe = p.clamp(min=0.0)
    return (1.0 - p_safe.pow(a).sum(dim=dim)) / (a - 1.0)
