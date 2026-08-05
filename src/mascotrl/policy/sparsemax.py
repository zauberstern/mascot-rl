"""Sparsemax: Euclidean projection onto the probability simplex.

Martins & Astudillo, ICML 2016 (Algorithm 1). No external dependency.
"""
from __future__ import annotations

import torch
from torch import Tensor


def sparsemax(z: Tensor, dim: int = -1) -> Tensor:
    """Project ``z`` onto the probability simplex along ``dim``.

    Unlike softmax (entropic projection), sparsemax can assign exact zeros
    and has a non-vanishing gradient near the simplex boundary for support
    coordinates.
    """
    sorted_z, _ = z.sort(dim=dim, descending=True)
    cumsum = sorted_z.cumsum(dim=dim)
    k = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.ndim
    shape[dim] = -1
    k = k.reshape(shape)
    support = (1 + k * sorted_z > cumsum).sum(dim=dim, keepdim=True).clamp_min(1)
    tau = (cumsum.gather(dim, support - 1) - 1) / support.to(dtype=z.dtype)
    return (z - tau).clamp(min=0)
