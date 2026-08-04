"""ADV / liquidity under dynamic universes: slot-align wide ADV to (T, K)."""
from __future__ import annotations

import numpy as np
import pytest

from src.features.blocks.liquidity import build_liquidity_block, map_wide_to_slots


def test_map_wide_to_slots_time_varying_occupancy():
    # Wide fingerprint: 3 names; slots K=2 with rotating occupants.
    wide_secids = [10, 20, 30]
    wide = np.array(
        [
            [100.0, 200.0, 300.0],
            [110.0, 210.0, 310.0],
            [120.0, 220.0, 320.0],
        ],
        dtype=np.float64,
    )
    slots_rows = [
        [10, 20],
        [10, 30],
        [20, 30],
    ]
    slotted = map_wide_to_slots(wide, secids=wide_secids, slots_rows=slots_rows)
    assert slotted.shape == (3, 2)
    np.testing.assert_allclose(slotted[0], [100.0, 200.0])
    np.testing.assert_allclose(slotted[1], [110.0, 310.0])
    np.testing.assert_allclose(slotted[2], [220.0, 320.0])


def test_build_liquidity_block_accepts_wide_adv_plus_slots():
    rng = np.random.default_rng(1)
    t, n, k = 40, 5, 2
    wide_rets = rng.normal(scale=0.01, size=(t, n))
    wide_dv = rng.uniform(1e5, 1e6, size=(t, n))
    secids = list(range(n))
    # Time-varying slot mask / occupancy.
    slots_rows = []
    for i in range(t):
        a = i % n
        b = (i + 1) % n
        slots_rows.append([a, b])
    # Slotted returns for the cube.
    rets = map_wide_to_slots(wide_rets, secids=secids, slots_rows=slots_rows)
    cube, names = build_liquidity_block(
        rets,
        wide_dollar_volume=wide_dv,
        wide_secids=secids,
        slots_rows=slots_rows,
    )
    assert names == ["amihud", "adv_dollar_volume"]
    assert cube.shape == (t, k, 2)
    assert np.isfinite(cube[30:]).any()


def test_build_liquidity_still_requires_matching_direct_dollar_volume():
    r = np.ones((10, 3))
    dv = np.ones((10, 4))
    with pytest.raises(ValueError, match="shape"):
        build_liquidity_block(r, dollar_volume=dv)
