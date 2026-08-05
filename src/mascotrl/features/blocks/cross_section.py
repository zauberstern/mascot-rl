"""Cross-sectional normalize law applied to feature cubes."""
from __future__ import annotations

import numpy as np

from mascotrl.features.blocks.normalize import normalize_cross_section_panel


def apply_cross_section_normalize(cube: np.ndarray) -> np.ndarray:
    """Apply winsorize→XS-z→clip independently per channel.

    ``cube`` shape ``(T, K, C)``.
    """
    arr = np.asarray(cube, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"expected (T, K, C), got {arr.shape}")
    t, k, c = arr.shape
    out = np.empty_like(arr)
    for i in range(c):
        out[:, :, i] = normalize_cross_section_panel(arr[:, :, i])
    return out
