"""Strict parity: pandas rolling adapters vs reference NumPy loops."""
from __future__ import annotations

import numpy as np

from src.features.blocks.pandas_rolling import (
    amihud_illiquidity_pandas,
    trailing_hv_panel_pandas,
)


def _hv_numpy(returns: np.ndarray, window: int) -> np.ndarray:
    ann = np.sqrt(252.0)
    r = np.asarray(returns, dtype=np.float64)
    t_len, k = r.shape
    w = int(window)
    out = np.full((t_len, k), np.nan, dtype=np.float64)
    for t in range(t_len):
        if t + 1 < w:
            continue
        block = r[t + 1 - w : t + 1]
        for j in range(k):
            col = block[:, j]
            finite = col[np.isfinite(col)]
            if finite.size < max(2, w // 2):
                continue
            out[t, j] = float(np.std(finite, ddof=1) * ann)
    return out


def _amihud_numpy(returns, dollar_volume, window=21):
    r = np.asarray(returns, dtype=np.float64)
    dv = np.asarray(dollar_volume, dtype=np.float64)
    t_len, k = r.shape
    w = int(window)
    out = np.full((t_len, k), np.nan, dtype=np.float64)
    safe_dv = np.where(dv > 0.0, dv, np.nan)
    ratio = np.abs(r) / safe_dv
    for t in range(t_len):
        if t + 1 < w:
            continue
        block = ratio[t + 1 - w : t + 1]
        for j in range(k):
            col = block[:, j]
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                continue
            out[t, j] = float(np.mean(finite))
    return out


def test_hv_pandas_matches_numpy_on_clean_panel():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, size=(80, 4))
    for w in (21, 63):
        a = _hv_numpy(r, w)
        b = trailing_hv_panel_pandas(r, w)
        assert np.allclose(a, b, equal_nan=True, rtol=1e-12, atol=1e-12)


def test_amihud_pandas_matches_numpy_on_clean_panel():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, size=(60, 3))
    dv = rng.uniform(1e6, 5e6, size=r.shape)
    a = _amihud_numpy(r, dv, window=21)
    b = amihud_illiquidity_pandas(r, dv, window=21)
    assert np.allclose(a, b, equal_nan=True, rtol=1e-12, atol=1e-12)


def test_public_api_uses_pandas_path():
    from src.features.blocks.liquidity import amihud_illiquidity
    from src.features.blocks.volatility_vrp import trailing_hv_panel

    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.01, size=(50, 2))
    dv = np.ones_like(r) * 1e6
    assert np.allclose(
        trailing_hv_panel(r, 21), trailing_hv_panel_pandas(r, 21), equal_nan=True
    )
    assert np.allclose(
        amihud_illiquidity(r, dv, window=21),
        amihud_illiquidity_pandas(r, dv, window=21),
        equal_nan=True,
    )
