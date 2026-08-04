"""Math fixtures for range-volatility estimators."""
from __future__ import annotations

import numpy as np
import pytest

from src.features.blocks.range_volatility import (
    GK_CONST,
    build_range_volatility_block,
    garman_klass_var_daily,
    parkinson_var_daily,
    rogers_satchell_var_daily,
    yang_zhang_k,
)


def test_garman_klass_worked_example() -> None:
    # Published 2-day example: day vars ~0.002034 / 0.002143 → mean ~0.002089
    o = np.array([[100.0], [103.0]])
    h = np.array([[105.0], [107.0]])
    l = np.array([[98.0], [100.0]])
    c = np.array([[103.0], [101.0]])
    v = garman_klass_var_daily(o, h, l, c).ravel()
    assert v[0] == pytest.approx(0.002034, abs=5e-5)
    assert v[1] == pytest.approx(0.002143, abs=5e-5)
    assert float(v.mean()) == pytest.approx(0.002089, abs=5e-5)
    assert GK_CONST == pytest.approx(2 * np.log(2) - 1)


def test_parkinson_constant_range_identity() -> None:
    ratio = 1.1
    h = np.full((5, 2), 110.0)
    l = np.full((5, 2), 100.0)
    expected = (np.log(ratio) ** 2) / (4.0 * np.log(2.0))
    v = parkinson_var_daily(h, l)
    np.testing.assert_allclose(v, expected, rtol=1e-10)


def test_rogers_satchell_finite_with_drift() -> None:
    t = np.arange(30, dtype=float)
    o = 100 * np.exp(0.01 * t)[:, None]
    c = o * 1.01
    h = np.maximum(o, c) * 1.02
    l = np.minimum(o, c) * 0.98
    v = rogers_satchell_var_daily(o, h, l, c)
    assert np.all(np.isfinite(v))
    assert np.all(v >= -1e-9)


def test_yang_zhang_k_formula() -> None:
    n = 21
    expected = 0.34 / (1.34 + (n + 1) / (n - 1))
    assert yang_zhang_k(n) == pytest.approx(expected)


def test_build_range_volatility_block_shapes() -> None:
    rng = np.random.default_rng(0)
    t, k = 60, 3
    close = 100 + np.cumsum(rng.normal(0, 0.5, size=(t, k)), axis=0)
    open_ = close + rng.normal(0, 0.1, size=(t, k))
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2
    ohlc = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "adj_close": close,
    }
    cube, names = build_range_volatility_block(ohlc, window=21)
    assert cube.shape == (t, k, 4)
    assert names == [
        "parkinson_21",
        "garman_klass_21",
        "rogers_satchell_21",
        "yang_zhang_21",
    ]
    assert np.isnan(cube[:20]).all()
    assert np.isfinite(cube[40:]).any()
