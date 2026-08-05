"""Kritzman-Chow turbulence index: causal rolling Mahalanobis + expanding threshold."""
from __future__ import annotations

import numpy as np
import pytest

from mascotrl.eval.turbulence import classify_regime, turbulence_index


def test_turbulence_causal_window_ignores_future_spike() -> None:
    rng = np.random.default_rng(0)
    t, n = 400, 6
    returns = rng.normal(0.0, 0.01, size=(t, n))
    # Inject a huge cross-sectional spike only at the last day.
    returns[-1, :] = 0.5
    d = turbulence_index(returns, window=252)
    # Day before the spike must not see future covariance/mean contamination.
    assert np.isfinite(d[-2])
    assert d[-2] < 50.0
    assert d[-1] > d[-2] * 5.0


def test_turbulence_uses_strictly_past_rows() -> None:
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0, 0.01, size=(300, 4))
    d_full = turbulence_index(returns, window=100)
    # Truncating the series after t must not change d[t] (causality).
    t = 200
    d_trunc = turbulence_index(returns[: t + 1], window=100)
    np.testing.assert_allclose(d_full[t], d_trunc[t], rtol=1e-10, atol=1e-12)


def test_turbulence_nan_before_warmup() -> None:
    returns = np.zeros((10, 3))
    d = turbulence_index(returns, window=50)
    assert np.all(np.isnan(d))


def test_classify_regime_expanding_quantile_no_lookahead() -> None:
    rng = np.random.default_rng(0)
    calm = 1.0 + 0.05 * rng.standard_normal(120)
    turb = np.concatenate([calm, np.full(25, 80.0), calm[:40]])
    labels = classify_regime(turb, quantile=0.75)
    assert labels.dtype == bool
    # Mid-calm (before any burst) should mostly be non-turbulent; spot-check.
    assert not labels[40]
    assert labels[130]  # inside injected burst
    # Causality: label at t depends only on turb[:t+1]. Recompute prefix.
    labels_pref = classify_regime(turb[:41], quantile=0.75)
    assert labels[40] == labels_pref[40]


def test_classify_regime_monotonic_in_turbulence() -> None:
    turb = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    labels = classify_regime(turb, quantile=0.5)
    # Once values climb above expanding median of past, flags should rise.
    assert labels[-1]
    assert labels.sum() >= 1


def test_turbulence_with_macro_cols_prefix_stable() -> None:
    """Optional PIT VIX/OAS/term columns: truncating future must not change d[t]."""
    rng = np.random.default_rng(2)
    t, n = 350, 5
    returns = rng.normal(0.0, 0.01, size=(t, n))
    # Pre-registered macro co-movement (already lagged / known at t).
    macro = np.column_stack(
        [
            15.0 + rng.normal(0, 1.0, t),  # vix
            4.0 + rng.normal(0, 0.2, t),  # hy_oas
            1.0 + rng.normal(0, 0.1, t),  # term_spread
        ]
    )
    d_full = turbulence_index(returns, window=80, macro_cols=macro)
    cut = 220
    d_pref = turbulence_index(
        returns[: cut + 1], window=80, macro_cols=macro[: cut + 1]
    )
    np.testing.assert_allclose(d_full[cut], d_pref[cut], rtol=1e-10, atol=1e-12)
    assert np.isfinite(d_full[cut])


def test_turbulence_macro_cols_length_mismatch_raises() -> None:
    returns = np.zeros((50, 3))
    macro = np.zeros((40, 2))
    with pytest.raises(ValueError, match="macro_cols"):
        turbulence_index(returns, window=20, macro_cols=macro)


def test_turbulence_scaled_macro_moves_dt_without_blowing_scale() -> None:
    """Raw VIX (~20) must not dominate returns (~0.01); windowed z-score only."""
    rng = np.random.default_rng(5)
    t, n = 200, 4
    returns = rng.normal(0.0, 0.01, size=(t, n))
    vix = np.full(t, 20.0, dtype=np.float64)
    vix[-10:] = 25.0
    macro = vix.reshape(-1, 1)
    d_scaled = turbulence_index(
        returns, window=80, macro_cols=macro, scale_macro=True, min_names=3
    )
    d_ret_only = turbulence_index(returns, window=80, min_names=3)
    assert np.isfinite(d_scaled[-1])
    # Scaled path reacts to the VIX spike vs returns-only.
    assert abs(d_scaled[-1] - d_ret_only[-1]) > 1e-6
    # Distance stays O(n) not O(1e4) from raw-level concat.
    assert d_scaled[-1] < 500.0


def test_turbulence_nan_name_not_treated_as_zero() -> None:
    """Missing return must not equal inventing a zero return."""
    rng = np.random.default_rng(9)
    t, n = 150, 5
    returns = rng.normal(0.0, 0.01, size=(t, n))
    returns_nan = returns.copy()
    returns_nan[-1, 0] = np.nan
    returns_zero = returns.copy()
    returns_zero[-1, 0] = 0.0
    d_nan = turbulence_index(returns_nan, window=60, min_names=3)
    d_zero = turbulence_index(returns_zero, window=60, min_names=3)
    assert np.isfinite(d_nan[-1])
    assert np.isfinite(d_zero[-1])
    assert d_nan[-1] != pytest.approx(d_zero[-1], rel=0, abs=1e-12)


def test_turbulence_too_few_finite_names_yields_nan() -> None:
    returns = np.full((100, 6), np.nan, dtype=np.float64)
    returns[:, 0] = 0.01
    d = turbulence_index(returns, window=40, min_names=5)
    assert np.isnan(d[-1])


def test_classify_regime_inclusive_of_dt_documented() -> None:
    """Expanding q at t includes d_t (causal because μ/Σ used only the past)."""
    turb = np.array([1.0, 1.0, 1.0, 100.0])
    labels = classify_regime(turb, quantile=0.75)
    assert labels[-1]
