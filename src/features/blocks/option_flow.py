"""Option flow + dividend yield observation channels."""
from __future__ import annotations

from typing import Mapping

import numpy as np


def _as_tk(arr: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be (T,K), got {a.shape}")
    return a


def _lag_diff(panel: np.ndarray, lag: int) -> np.ndarray:
    out = np.full_like(panel, np.nan, dtype=np.float64)
    if lag <= 0:
        return out
    out[lag:] = panel[lag:] - panel[:-lag]
    return out


def build_option_flow_block(
    option_flow: Mapping[str, np.ndarray] | None,
) -> tuple[np.ndarray, list[str]]:
    if option_flow is None:
        return np.zeros((0, 0, 0), dtype=np.float64), []
    channels: list[np.ndarray] = []
    names: list[str] = []
    for key in ("pc_vol", "pc_oi", "opt_stock_vol"):
        if key in option_flow:
            channels.append(_as_tk(option_flow[key], key))
            names.append(key)
    if "oi_lvl" in option_flow:
        oi = _as_tk(option_flow["oi_lvl"], "oi_lvl")
        channels.append(_lag_diff(oi, 21))
        names.append("oi_chg_21")
    if "div_yield_ttm" in option_flow:
        channels.append(_as_tk(option_flow["div_yield_ttm"], "div_yield_ttm"))
        names.append("div_yield_ttm")
    if not channels:
        return np.zeros((0, 0, 0), dtype=np.float64), []
    return np.stack(channels, axis=-1), names
