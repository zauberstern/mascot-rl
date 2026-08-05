"""Microstructure feature block: effective spread, VWAP deviation, block share, turnover."""
from __future__ import annotations

import numpy as np


def _causal_mean(panel: np.ndarray, window: int) -> np.ndarray:
    t_len, k = panel.shape
    w = int(window)
    out = np.full((t_len, k), np.nan, dtype=np.float64)
    for t in range(t_len):
        if t + 1 < w:
            continue
        block = panel[t + 1 - w : t + 1]
        for j in range(k):
            col = block[:, j]
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                continue
            out[t, j] = float(np.mean(finite))
    return out


def build_microstructure_block(
    microstructure: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, list[str]]:
    if microstructure is None:
        return np.zeros((0, 0, 0), dtype=np.float64), []
    keys = ("eff_spread", "vwap_dev", "block_share", "turnover")
    missing = [k for k in keys if k not in microstructure]
    if missing:
        raise ValueError(f"microstructure missing keys {missing}")
    eff = np.asarray(microstructure["eff_spread"], dtype=np.float64)
    vwap = np.asarray(microstructure["vwap_dev"], dtype=np.float64)
    blk = np.asarray(microstructure["block_share"], dtype=np.float64)
    turn = np.asarray(microstructure["turnover"], dtype=np.float64)
    channels = [
        _causal_mean(eff, 21),
        _causal_mean(vwap, 5),
        _causal_mean(blk, 21),
        _causal_mean(turn, 21),
    ]
    cube = np.stack(channels, axis=-1)
    return cube, ["eff_spread_21", "vwap_dev_5", "block_share_21", "turnover_21"]
