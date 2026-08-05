"""Winsorize / z-score helpers (causal expanding + cross-sectional)."""
from __future__ import annotations

import numpy as np


def winsorize_panel(
    x: np.ndarray,
    *,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> np.ndarray:
    """Deprecated global winsorize; prefer :func:`winsorize_cross_section` (L10)."""
    arr = np.asarray(x, dtype=np.float64).copy()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    lo = float(np.quantile(finite, lower_q))
    hi = float(np.quantile(finite, upper_q))
    return np.clip(arr, lo, hi)


def winsorize_cross_section(
    x: np.ndarray,
    *,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> np.ndarray:
    """Per-date (row) winsorize at cross-sectional quantiles (Phase L10 step 1)."""
    arr = np.asarray(x, dtype=np.float64).copy()
    if arr.ndim != 2:
        raise ValueError(f"expected (T, K), got {arr.shape}")
    for t in range(arr.shape[0]):
        row = arr[t]
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            continue
        lo = float(np.quantile(finite, lower_q))
        hi = float(np.quantile(finite, upper_q))
        arr[t] = np.clip(row, lo, hi)
    return arr


def cross_sectional_zscore(x: np.ndarray, *, clip: float = 3.0) -> np.ndarray:
    """Per-date (row) cross-sectional z-score; clip to ±clip.

    ``x`` shape ``(T, K)``. Mean/std computed across names at each date.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected (T, K), got {arr.shape}")
    mu = np.nanmean(arr, axis=1, keepdims=True)
    sd = np.nanstd(arr, axis=1, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    z = (arr - mu) / sd
    return np.clip(z, -float(clip), float(clip))


def expanding_causal_zscore(x: np.ndarray, *, clip: float = 3.0, min_obs: int = 2) -> np.ndarray:
    """Expanding z-score per column using only observations at indices ``<= t``.

    No lookahead: ``z[t]`` depends solely on ``x[:t+1]``. Early rows with
    fewer than ``min_obs`` finite points are NaN.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected (T, K), got {arr.shape}")
    t_len, k = arr.shape
    out = np.full_like(arr, np.nan, dtype=np.float64)
    for j in range(k):
        col = arr[:, j]
        for t in range(t_len):
            window = col[: t + 1]
            finite = window[np.isfinite(window)]
            if finite.size < int(min_obs):
                continue
            mu = float(np.mean(finite))
            sd = float(np.std(finite))
            if sd < 1e-12:
                out[t, j] = 0.0
            else:
                out[t, j] = (col[t] - mu) / sd
    return np.clip(out, -float(clip), float(clip))


def normalize_cross_section_panel(
    x: np.ndarray,
    *,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    clip: float = 3.0,
) -> np.ndarray:
    """L10 law: per-date winsorize → cross-sectional z-score → ±clip."""
    w = winsorize_cross_section(x, lower_q=lower_q, upper_q=upper_q)
    return cross_sectional_zscore(w, clip=clip)
