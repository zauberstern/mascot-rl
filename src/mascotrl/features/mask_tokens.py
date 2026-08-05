"""EarnMore-style maskable representations and mask honesty helpers."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.policy.rasp_locks import assert_mask_honesty


class MaskTokenEncoder(nn.Module):
    """Replace invalid slot feature vectors with a learned mask token.

    Zhang 2023 EarnMore-style: masking acts on the representation, not only
    on output logits.
    """

    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.n_channels = int(n_channels)
        self.mask_token = nn.Parameter(torch.zeros(self.n_channels))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``features`` (..., K, C), ``mask`` (..., K) with 1=valid."""
        m = mask.to(dtype=features.dtype, device=features.device)
        while m.dim() < features.dim():
            m = m.unsqueeze(-1)
        token = self.mask_token.to(device=features.device, dtype=features.dtype)
        token_b = token.view(*([1] * (features.dim() - 1)), self.n_channels)
        return features * m + token_b * (1.0 - m)


def apply_mask_token(
    features: torch.Tensor,
    mask: torch.Tensor,
    mask_token: torch.Tensor,
) -> torch.Tensor:
    """Functional form used in tests and non-module call sites."""
    m = mask.to(dtype=features.dtype, device=features.device)
    while m.dim() < features.dim():
        m = m.unsqueeze(-1)
    token = mask_token.to(device=features.device, dtype=features.dtype)
    token_b = token.view(*([1] * (features.dim() - 1)), token.shape[-1])
    return features * m + token_b * (1.0 - m)


def apply_mask_tokens_to_cube(
    cube: np.ndarray,
    mask: np.ndarray,
    *,
    token: np.ndarray | None = None,
) -> np.ndarray:
    """Replace invalid slots in a ``(T,K,C)`` or ``(K,C)`` cube with a token."""
    x = np.asarray(cube, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    if x.ndim == 2:
        if m.ndim != 1 or m.shape[0] != x.shape[0]:
            raise ValueError(f"mask shape {m.shape} incompatible with cube {x.shape}")
        tok = (
            np.zeros(x.shape[-1], dtype=np.float64)
            if token is None
            else np.asarray(token, dtype=np.float64)
        )
        out = x.copy()
        out[m <= 0.5] = tok
        return out
    if x.ndim == 3:
        if m.ndim == 1:
            m2 = np.broadcast_to(m[None, :], (x.shape[0], x.shape[1]))
        else:
            m2 = m
        if m2.shape[:2] != x.shape[:2]:
            raise ValueError(f"mask shape {m.shape} incompatible with cube {x.shape}")
        tok = (
            np.zeros(x.shape[-1], dtype=np.float64)
            if token is None
            else np.asarray(token, dtype=np.float64)
        )
        out = x.copy()
        out[m2 <= 0.5] = tok
        return out
    raise ValueError(f"cube ndim must be 2 or 3, got {x.ndim}")


def refuse_logit_only_masking(*, representation_masked: bool) -> None:
    if not representation_masked:
        raise ValueError(
            "logit_only_masking_refused: EarnMore-style masking must act on "
            "the representation (mask tokens), not only on output logits"
        )


__all__ = [
    "MaskTokenEncoder",
    "apply_mask_token",
    "apply_mask_tokens_to_cube",
    "assert_mask_honesty",
    "refuse_logit_only_masking",
]
