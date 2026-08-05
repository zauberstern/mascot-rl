"""Shared CNN encoder over Kelly-style IV surface images (11 x 34)."""
from __future__ import annotations

import torch
import torch.nn as nn


class SurfaceImageEncoder(nn.Module):
    """Weight-shared CNN: ``(B, 1, 11, 34) → (B, embed_dim)``.

    Kelly 2026 locality argument: convolutional filters over the OM
    delta-tenor grid beat unstructured MLPs on the same pixels.
    """

    def __init__(self, *, embed_dim: int = 16) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, self.embed_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(f"expected (B,1,H,W) or (B,H,W), got {tuple(x.shape)}")
        return self.net(x)
