"""E-2: hand-computed Amihud illiquidity and ADV rolling window (window=21 default)."""
from __future__ import annotations

import numpy as np
import pytest

from mascotrl.features.blocks.liquidity import amihud_illiquidity, build_liquidity_block

WINDOW = 3  # small window for hand-check; production default is 21


def test_amihud_exact_ratio_mean_over_window() -> None:
    """Amihud at t is mean(|r|/dollar_volume) over the trailing causal window."""
    r = np.array([[0.01, 0.02], [-0.02, 0.01], [0.03, -0.01]], dtype=np.float64)
    dv = np.array([[1e6, 2e6], [2e6, 3e6], [4e6, 5e6]], dtype=np.float64)
    ami = amihud_illiquidity(r, dv, window=WINDOW)
    # t=2, window=3: mean of |r|/dv rows 0..2 per column.
    col0 = np.mean([0.01 / 1e6, 0.02 / 2e6, 0.03 / 4e6])
    col1 = np.mean([0.02 / 2e6, 0.01 / 3e6, 0.01 / 5e6])
    assert ami[0, 0] != ami[0, 0]  # NaN until window-1
    np.testing.assert_allclose(ami[2, 0], col0, rtol=0, atol=1e-15)
    np.testing.assert_allclose(ami[2, 1], col1, rtol=0, atol=1e-15)


def test_adv_trailing_mean_dollar_volume_window_documented() -> None:
    """ADV is trailing mean dollar volume; window length matches Amihud (21 prod, 3 here)."""
    r = np.array([[0.01], [-0.02], [0.03]], dtype=np.float64)
    dv = np.array([[1e6], [2e6], [4e6]], dtype=np.float64)
    cube, names = build_liquidity_block(r, dollar_volume=dv, window=WINDOW)
    assert names == ["amihud", "adv_dollar_volume"]
    expected_adv = np.mean(dv[:WINDOW, 0])
    assert cube[WINDOW - 1, 0, 1] == pytest.approx(expected_adv)
    assert WINDOW == 3  # documented fixture window; production uses window=21
