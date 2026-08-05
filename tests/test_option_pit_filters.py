"""Step 8: option PIT filters (t−1 bid/ask/OI/vol)."""
from __future__ import annotations

import pandas as pd
import pytest

from mascotrl.features.pit_universe import option_pit_filter_mask


def test_option_pit_filters_spread_and_liquidity():
    df = pd.DataFrame(
        {
            "bid": [1.0, 2.0, 0.5, 1.0],
            "ask": [1.2, 1.5, 0.6, 1.1],  # row1 ask<bid → drop
            "volume": [10, 10, 0, 10],
            "open_interest": [100, 100, 50, 0],
            "delta": [0.5, 0.4, 0.3, 0.2],
            "expiry": ["2020-06-01"] * 4,
        }
    )
    keep = option_pit_filter_mask(df, asof="2020-01-01", min_volume=1.0, min_oi=1.0)
    assert keep.tolist() == [True, False, False, False]


def test_option_pit_missing_column_raises():
    df = pd.DataFrame({"bid": [1.0], "ask": [1.1]})
    with pytest.raises(ValueError, match="missing required"):
        option_pit_filter_mask(df, asof="2020-01-01")
