"""assemble_regime_desk NaN-safe turbulence path."""
from __future__ import annotations

import numpy as np

from mascotrl.eval.turbulence import turbulence_index


def test_desk_panel_nan_not_identical_to_zero_fill() -> None:
    """NaN names must not match the Mahalanobis of an explicit zero-fill."""
    rng = np.random.default_rng(0)
    t, n = 120, 6
    panel = rng.normal(0, 0.01, size=(t, n))
    panel_nan = panel.copy()
    panel_nan[80:, 0] = np.nan
    panel_zero = panel_nan.copy()
    panel_zero[~np.isfinite(panel_zero)] = 0.0
    d_nan = turbulence_index(panel_nan, window=40, min_names=3)
    d_zero = turbulence_index(panel_zero, window=40, min_names=3)
    # At least one post-warmup day differs (NaN drop vs fake zero co-move).
    finite = np.isfinite(d_nan) & np.isfinite(d_zero)
    assert finite.any()
    assert not np.allclose(d_nan[finite], d_zero[finite])
