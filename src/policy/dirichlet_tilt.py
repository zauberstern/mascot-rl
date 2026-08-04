"""Dirichlet-tilt action law (RASP Rank-1 head).

Law (plan A.3.4):
1. alpha = softplus(f) + eps
2. u ~ Dir(alpha) (train) or mean (eval)
3. log pi = Dir(alpha).log_prob(u)
4. w_prop = normalize( (w_base * (1 + kappa * (u - u_bar)))_+ ) with mask
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

DIRICHLET_EPS = 1e-3
DIRICHLET_TILT_HEADS = frozenset(
    {"dirichlet_tilt", "dirichlet_mean", "dirichlet_entropy"}
)


def concentrations_from_logits(raw: torch.Tensor, *, eps: float = DIRICHLET_EPS) -> torch.Tensor:
    """Map unbounded actor logits to Dirichlet concentrations."""
    return F.softplus(raw) + float(eps)


def dirichlet_sample(
    alpha: torch.Tensor,
    *,
    deterministic: bool = False,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample (or take mean) from Dir(alpha).

    When ``mask`` is provided, inactive slots are dropped from the support
    (plan A.3.4): their mass is zeroed and the remaining weights renormalised.
    """
    alpha = alpha.clamp_min(1e-4)
    if mask is not None:
        m = mask.to(dtype=alpha.dtype, device=alpha.device)
        if m.dim() == 1:
            m = m.unsqueeze(0).expand_as(alpha)
        m = (m > 0.5).to(dtype=alpha.dtype)
        alpha = torch.where(m > 0.5, alpha, torch.full_like(alpha, 1e-6))
    dist = torch.distributions.Dirichlet(alpha)
    if deterministic:
        u = alpha / alpha.sum(dim=-1, keepdim=True)
    else:
        u = dist.rsample() if dist.has_rsample else dist.sample()
    if mask is not None:
        u = u * m
        u = u / u.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    logp = dist.log_prob(u.clamp_min(1e-8))
    ent = dist.entropy()
    return u, logp, ent


def dirichlet_log_prob(alpha: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    alpha = alpha.clamp_min(1e-4)
    u = u.clamp_min(1e-8)
    # Renormalise u onto the simplex in case of float drift.
    u = u / u.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.distributions.Dirichlet(alpha).log_prob(u)


def dirichlet_entropy(alpha: torch.Tensor) -> torch.Tensor:
    return torch.distributions.Dirichlet(alpha.clamp_min(1e-4)).entropy()


def multiplicative_tilt(
    u: torch.Tensor,
    *,
    w_base: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Map Dirichlet sample ``u`` to a long-only proposal around ``w_base``.

    When ``u == u_bar`` (uniform on the active support), ``w_prop == w_base``.
    """
    if u.dim() == 1:
        u = u.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    k = int(u.shape[-1])
    if mask is None:
        m = torch.ones_like(u)
    else:
        m = mask.to(dtype=u.dtype, device=u.device)
        if m.dim() == 1:
            m = m.unsqueeze(0).expand_as(u)
        m = (m > 0.5).to(dtype=u.dtype)

    active = m.sum(dim=-1, keepdim=True).clamp_min(1.0)
    if w_base is None:
        wb = m / active
    else:
        wb = w_base.to(dtype=u.dtype, device=u.device)
        if wb.dim() == 1:
            wb = wb.unsqueeze(0).expand_as(u)
        wb = wb * m
        wb = wb / wb.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    u_bar = m / active
    # Only tilt on active slots; inactive stay at 0.
    tilted = wb * (1.0 + float(kappa) * (u - u_bar))
    tilted = torch.clamp(tilted, min=0.0) * m
    w = tilted / tilted.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return w.squeeze(0) if squeeze else w


def apply_dirichlet_tilt_head(
    raw_or_u: torch.Tensor,
    *,
    mode: str = "from_u",
    w_base: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    kappa: float = 1.0,
    deterministic: bool = True,
) -> torch.Tensor:
    """Apply the dirichlet_tilt head.

    ``mode='from_u'``: ``raw_or_u`` is already a simplex sample (stored action).
    ``mode='from_logits'``: ``raw_or_u`` is actor logits; sample/mean then tilt.
    """
    key = str(mode).lower()
    if key == "from_logits":
        alpha = concentrations_from_logits(raw_or_u)
        u, _, _ = dirichlet_sample(alpha, deterministic=deterministic)
    elif key == "from_u":
        u = raw_or_u
    else:
        raise ValueError(f"unknown dirichlet tilt mode={mode!r}")
    return multiplicative_tilt(u, w_base=w_base, mask=mask, kappa=kappa)
