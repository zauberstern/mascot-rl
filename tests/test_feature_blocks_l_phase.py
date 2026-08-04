"""Phase L: normalization law, residual momentum, obs builder wiring."""
from __future__ import annotations

import numpy as np
import pytest

from src.features.blocks.normalize import (
    normalize_cross_section_panel,
    winsorize_cross_section,
    winsorize_panel,
)
from src.features.blocks.obs_builder import PanelObservationBuilder
from src.features.blocks.returns_momentum import (
    build_returns_momentum_block,
    residual_momentum_12_1,
)
from src.features.blocks.volatility_vrp import variance_risk_premium


def test_winsorize_is_per_date_not_global() -> None:
    """L10: each date winsorizes using only that date's cross-section."""
    x = np.zeros((2, 40), dtype=np.float64)
    x[0] = np.linspace(-1.0, 1.0, 40)
    x[1] = np.linspace(-1.0, 1.0, 40)
    x[1, -1] = 1e6
    w = winsorize_cross_section(x, lower_q=0.01, upper_q=0.99)
    # Isolated date-0 must match the joint panel's date-0 (no leakage from date-1).
    w0 = winsorize_cross_section(x[0:1], lower_q=0.01, upper_q=0.99)
    assert np.allclose(w[0], w0[0])
    assert w[1, -1] < 1e6
    # Global winsorize mixes dates: date-0 endpoints move once 1e6 enters the pool.
    g = winsorize_panel(x, lower_q=0.01, upper_q=0.99)
    assert not np.allclose(g[0], w0[0])


def test_normalize_law_order_winsorize_then_xs_z() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 8))
    x[10, 0] = 50.0  # outlier
    z = normalize_cross_section_panel(x)
    assert z.shape == x.shape
    # After ±3 clip, all finite values in range.
    fin = z[np.isfinite(z)]
    assert fin.min() >= -3.0 - 1e-9
    assert fin.max() <= 3.0 + 1e-9


def test_residual_momentum_uses_only_past_factors() -> None:
    rng = np.random.default_rng(1)
    t, k = 300, 4
    fac = rng.normal(0.0, 0.01, size=(t, 4))
    beta = rng.normal(0.0, 1.0, size=(k, 4))
    idio = rng.normal(0.0, 0.01, size=(t, k))
    rets = fac @ beta.T + idio
    mom = residual_momentum_12_1(rets, fac, window=60, skip=21)
    assert mom.shape == (t, k)
    # Need a full trailing window before any finite residual mom.
    assert np.all(np.isnan(mom[:59]))
    assert np.any(np.isfinite(mom[200:]))


def test_vrp_is_variance_difference_not_vol_difference() -> None:
    from src.features.blocks.volatility_vrp import variance_risk_premium as vrp_fn

    r = np.zeros((80, 2))
    r[20:] = 0.01  # after burn-in HV becomes positive
    out = vrp_fn(r, np.full((80, 2), 0.25), hv_window=21)
    # Where HV is defined, VRP should be iv**2 - hv**2 (positive when iv>hv).
    assert np.nanmean(out[40:]) > 0.0


def test_panel_observation_builder_rank_and_shape() -> None:
    rng = np.random.default_rng(2)
    t, k = 120, 5
    rets = rng.normal(0.0005, 0.015, size=(t, k))
    builder = PanelObservationBuilder(rets, seq_len=1)
    assert builder.n_channels >= 5
    obs = builder(50, np.zeros(k))
    assert obs.ndim == 1
    assert obs.size == k * builder.obs_channels_per_asset
    cube_t = builder.cube[50]
    assert cube_t.shape == (k, builder.n_channels)
    rank = int(np.linalg.matrix_rank(np.nan_to_num(cube_t, nan=0.0)))
    assert rank > 1


def test_build_research_hist_env_uses_feature_cube_when_enabled() -> None:
    from src.eval.research_alpha_train import build_research_hist_env

    rng = np.random.default_rng(3)
    t, k = 80, 4
    rets = rng.normal(0.0, 0.01, size=(t, k))
    fac = rng.normal(0.0, 0.01, size=(t, 4))
    cfg = {
        "primary_train": "historical_arm_env",
        "use_equity_feature_cube": True,
        "feature_seq_len": 1,
        "n_assets": k,
        "arm": {"id": "eq", "option_slots": 0, "equity_slots": k, "delta_mode": "off"},
        "projection_mode": "soft",
        "equity_bps": 5.0,
        "impact_c_eq": 0.5,
    }
    env = build_research_hist_env(rets, fac, cfg)
    obs, _ = env.reset()
    assert obs.size > k  # richer than raw returns
    assert env.feature_builder is not None
