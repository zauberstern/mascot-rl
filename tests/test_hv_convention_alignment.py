"""HV formula: Arctic equity-panel path matches cube annualized sample stdev."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.equity_panel import _rolling_hv
from src.features.blocks.volatility_vrp import trailing_hv_panel


def test_equity_panel_hv_matches_cube_annualized_stdev_on_toy_series():
    rng = np.random.default_rng(0)
    r = rng.normal(loc=0.0, scale=0.01, size=60)
    # Panel path (pandas Series).
    s = pd.Series(r)
    panel_hv = _rolling_hv(s, 21).to_numpy(dtype=float)
    # Cube path (T, K=1).
    cube_hv = trailing_hv_panel(r.reshape(-1, 1), 21)[:, 0]
    # Warm-up rows are NaN on both; compare the rest.
    mask = np.isfinite(panel_hv) & np.isfinite(cube_hv)
    assert mask.sum() >= 30
    np.testing.assert_allclose(panel_hv[mask], cube_hv[mask], rtol=1e-10, atol=1e-12)
