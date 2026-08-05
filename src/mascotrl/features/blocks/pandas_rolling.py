"""Pandas rolling adapters for causal (T, K) feature panels."""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = np.sqrt(252.0)


def trailing_hv_panel_pandas(returns: np.ndarray, window: int) -> np.ndarray:
    """Annualized trailing stdev via ``DataFrame.rolling`` (causal)."""
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError(f"returns must be (T, K), got {r.shape}")
    w = int(window)
    if w <= 0:
        raise ValueError("window must be positive")
    min_p = max(2, w // 2)
    df = pd.DataFrame(r)
    out = df.rolling(window=w, min_periods=min_p).std(ddof=1) * ANN
    # Match hand-rolled: no estimate until full window length elapsed.
    arr = out.to_numpy(dtype=np.float64)
    if arr.shape[0] >= w:
        arr[: w - 1] = np.nan
    else:
        arr[:] = np.nan
    return arr


def amihud_illiquidity_pandas(
    returns: np.ndarray,
    dollar_volume: np.ndarray,
    *,
    window: int = 21,
) -> np.ndarray:
    """Amihud mean(|r|/dv) via pandas rolling mean (causal)."""
    r = np.asarray(returns, dtype=np.float64)
    dv = np.asarray(dollar_volume, dtype=np.float64)
    if r.shape != dv.shape or r.ndim != 2:
        raise ValueError("returns and dollar_volume must share shape (T, K)")
    w = int(window)
    safe_dv = np.where(dv > 0.0, dv, np.nan)
    ratio = np.abs(r) / safe_dv
    arr = (
        pd.DataFrame(ratio)
        .rolling(window=w, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float64)
    )
    if arr.shape[0] >= w:
        arr[: w - 1] = np.nan
    else:
        arr[:] = np.nan
    return arr


def trailing_mean_panel_pandas(x: np.ndarray, window: int) -> np.ndarray:
    """Causal trailing mean; NaN until full window."""
    arr_in = np.asarray(x, dtype=np.float64)
    if arr_in.ndim != 2:
        raise ValueError(f"x must be (T, K), got {arr_in.shape}")
    w = int(window)
    arr = (
        pd.DataFrame(arr_in)
        .rolling(window=w, min_periods=w)
        .mean()
        .to_numpy(dtype=np.float64)
    )
    return arr
