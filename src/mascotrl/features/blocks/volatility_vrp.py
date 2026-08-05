"""Historical volatility, vol-of-vol, optional VRP from IV/HV."""
from __future__ import annotations

import numpy as np

HV_WINDOWS: tuple[int, ...] = (21, 63, 252)
VOV_WINDOW = 21
ANN = np.sqrt(252.0)


def _as_tk(returns: np.ndarray) -> np.ndarray:
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError(f"returns must be (T, K), got {r.shape}")
    return r


def trailing_hv_panel(returns: np.ndarray, window: int) -> np.ndarray:
    """Annualized trailing stdev of returns; causal ``(T, K)``.

    Implemented via pandas rolling (library path) with the same full-window
    causal gate as the historical NumPy loop.
    """
    from src.features.blocks.pandas_rolling import trailing_hv_panel_pandas

    return trailing_hv_panel_pandas(returns, window)


def vol_of_vol(returns: np.ndarray, *, hv_window: int = 21, vov_window: int = VOV_WINDOW) -> np.ndarray:
    """Std of trailing HV over a causal window (vol-of-vol)."""
    hv = trailing_hv_panel(returns, hv_window)
    t_len, k = hv.shape
    out = np.full((t_len, k), np.nan, dtype=np.float64)
    w = int(vov_window)
    for t in range(t_len):
        if t + 1 < w:
            continue
        block = hv[t + 1 - w : t + 1]
        for j in range(k):
            col = block[:, j]
            finite = col[np.isfinite(col)]
            if finite.size < max(2, w // 2):
                continue
            out[t, j] = float(np.std(finite, ddof=1))
    return out


def variance_risk_premium(
    returns: np.ndarray,
    iv: np.ndarray | None,
    *,
    hv_window: int = 21,
) -> np.ndarray:
    """VRP = IV^2 - HV^2 when IV provided; else all-NaN (Phase L3)."""
    hv = trailing_hv_panel(returns, hv_window)
    if iv is None:
        return np.full_like(hv, np.nan)
    iv_arr = np.asarray(iv, dtype=np.float64)
    if iv_arr.shape != hv.shape:
        raise ValueError(f"iv shape {iv_arr.shape} != returns-derived HV {hv.shape}")
    return np.square(iv_arr) - np.square(hv)


def build_volatility_vrp_block(
    returns: np.ndarray,
    *,
    iv: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """HV windows + vol-of-vol + optional VRP → ``(T, K, C)``."""
    channels: list[np.ndarray] = []
    names: list[str] = []
    for w in HV_WINDOWS:
        channels.append(trailing_hv_panel(returns, w))
        names.append(f"hv_{w}")
    channels.append(vol_of_vol(returns))
    names.append("vol_of_vol")
    if iv is not None:
        channels.append(variance_risk_premium(returns, iv))
        names.append("vrp")
    cube = np.stack(channels, axis=-1)
    return cube, names


def vrp_30_level(mfiv_30: np.ndarray, hv_21: np.ndarray) -> np.ndarray:
    """Catalog VRP: ``mfiv_30 - hv_21`` (level difference; NaN if either leg missing)."""
    iv = np.asarray(mfiv_30, dtype=np.float64)
    hv = np.asarray(hv_21, dtype=np.float64)
    if iv.shape != hv.shape:
        raise ValueError(f"mfiv_30 shape {iv.shape} != hv_21 shape {hv.shape}")
    out = iv - hv
    out[~(np.isfinite(iv) & np.isfinite(hv))] = np.nan
    return out

