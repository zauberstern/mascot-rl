"""Phase 4: future-proof surface catalog features (shape, PIT ffill, NaN policy)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.surface_signals import (
    SURFACE_SIGNAL_NAMES,
    compute_surface_signals_panel,
)


def _grid_row(secid, date, days, delta, cp, iv):
    return {
        "secid": secid,
        "date": pd.Timestamp(date),
        "days": days,
        "delta": delta,
        "cp_flag": cp,
        "impl_volatility": iv,
        "impl_strike": 100.0,
        "impl_premium": 1.0,
        "dispersion": 0.01,
    }


def _daily_surface_with_geometry(n_days: int = 10) -> pd.DataFrame:
    """Minimal daily surface so term-slope / skew are finite and 5d diffs exist."""
    rows = []
    base = pd.Timestamp("2020-01-02")
    for i in range(n_days):
        d = (base + pd.DateOffset(days=int(i))).strftime("%Y-%m-%d")
        skew_put = 0.30 + 0.01 * i
        atm = 0.20 + 0.001 * i
        term = 0.25 + 0.002 * i
        rows.extend(
            [
                _grid_row(101, d, 30, -20, "P", skew_put),
                _grid_row(101, d, 30, 50, "C", atm),
                _grid_row(101, d, 365, 50, "C", term),
            ]
        )
    return pd.DataFrame(rows)


def test_catalog_includes_phase4_future_names():
    for name in ("d_iv_term_slope_5d", "d_iv_skew_5d", "vrp_30"):
        assert name in SURFACE_SIGNAL_NAMES


def test_5d_geometry_diffs_shape_and_pit():
    surf = _daily_surface_with_geometry(10)
    panel = compute_surface_signals_panel(surf, month_end_only=False)
    assert panel.shape[0] == 10
    assert "d_iv_term_slope_5d" in panel.columns
    assert "d_iv_skew_5d" in panel.columns
    assert panel["d_iv_term_slope_5d"].iloc[:5].isna().all()
    assert panel["d_iv_skew_5d"].iloc[:5].isna().all()
    assert np.isfinite(panel["d_iv_term_slope_5d"].iloc[5:]).all()
    assert np.isfinite(panel["d_iv_skew_5d"].iloc[5:]).all()
    term = panel["iv_term_slope"].to_numpy(dtype=float)
    d5 = panel["d_iv_term_slope_5d"].to_numpy(dtype=float)
    np.testing.assert_allclose(d5[5:], term[5:] - term[:-5], equal_nan=True)


def test_vrp_30_is_mfiv_minus_hv_and_nan_when_hv_missing():
    surf = _daily_surface_with_geometry(1)
    panel_no_hv = compute_surface_signals_panel(surf, month_end_only=False)
    assert np.isnan(panel_no_hv.iloc[0]["vrp_30"])
    date = panel_no_hv.iloc[0]["date"]
    hv = pd.DataFrame({"secid": [101], "date": [date], "hv": [0.15]})
    panel = compute_surface_signals_panel(surf, hv=hv, month_end_only=False)
    row = panel.iloc[0]
    # Sparse toy grids often leave mfiv NaN; never zero-fill VRP.
    if np.isfinite(row["mfiv_30"]):
        assert row["vrp_30"] == pytest.approx(float(row["mfiv_30"]) - 0.15)
    else:
        assert np.isnan(row["vrp_30"])


def test_vrp_30_level_helper_matches_catalog_definition():
    from src.features.blocks.volatility_vrp import vrp_30_level

    mfiv = np.array([[0.25, np.nan], [0.30, 0.40]])
    hv = np.array([[0.10, 0.10], [0.20, np.nan]])
    out = vrp_30_level(mfiv, hv)
    assert out[0, 0] == pytest.approx(0.15)
    assert np.isnan(out[0, 1]) and np.isnan(out[1, 1])
    assert out[1, 0] == pytest.approx(0.10)


def test_5d_diffs_do_not_bfill_or_zero_fill_gaps():
    rows = []
    for d, atm in (("2020-01-02", 0.20), ("2020-01-12", 0.22)):
        rows.extend(
            [
                _grid_row(101, d, 30, -20, "P", 0.30),
                _grid_row(101, d, 30, 50, "C", atm),
                _grid_row(101, d, 365, 50, "C", atm + 0.05),
            ]
        )
    panel = compute_surface_signals_panel(pd.DataFrame(rows), month_end_only=False)
    assert panel["d_iv_term_slope_5d"].isna().all()
    assert not (panel["d_iv_term_slope_5d"] == 0.0).any()
