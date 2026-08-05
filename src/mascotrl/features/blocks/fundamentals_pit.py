"""PIT fundamentals stacker: worldscope + IBES ratios + Compustat (already lagged)."""
from __future__ import annotations

from typing import Mapping

import numpy as np

# Canonical channel order (20 channels when all present).
FUNDAMENTALS_PIT_CHANNELS: tuple[str, ...] = (
    "bp",
    "ep",
    "ta_growth",
    "rev_growth",
    "bm",
    "ep_exi",
    "ps",
    "pcf",
    "dpr",
    "npm",
    "gpm",
    "roa",
    "roe",
    "cfm",
    "evm",
    "capex_inv",
    "at_growth",
    "sale_growth",
    "ni_at",
    "dvc_at",
)


def build_fundamentals_pit_block(
    fundamentals_pit: Mapping[str, np.ndarray] | None,
) -> tuple[np.ndarray, list[str]]:
    if fundamentals_pit is None:
        return np.zeros((0, 0, 0), dtype=np.float64), []
    cols: list[np.ndarray] = []
    names: list[str] = []
    shape: tuple[int, int] | None = None
    for name in FUNDAMENTALS_PIT_CHANNELS:
        arr = fundamentals_pit.get(name)
        if arr is None:
            continue
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim != 2:
            raise ValueError(f"fundamentals_pit[{name!r}] must be (T,K), got {a.shape}")
        if shape is None:
            shape = a.shape
        elif a.shape != shape:
            raise ValueError(
                f"fundamentals_pit[{name!r}] shape {a.shape} != {shape}"
            )
        cols.append(a)
        names.append(name)
    if not cols:
        return np.zeros((0, 0, 0), dtype=np.float64), []
    return np.stack(cols, axis=-1), names
