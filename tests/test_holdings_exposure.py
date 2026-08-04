"""Holdings-based characteristic exposures (DGTW-style)."""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import FLOAT_TOL


def test_holdings_exposures_hand_computed():
    from src.reporting.holdings_exposure import holdings_exposures

    # T=2, K=2
    weights = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float64)
    chars = {
        "log_mktcap": np.array([[2.0, 4.0], [2.0, 4.0]]),
        "book_to_market": np.array([[1.0, 0.0], [1.0, 0.0]]),
        "momentum_12_1": np.array([[0.1, -0.1], [0.1, -0.1]]),
        "roe": np.array([[0.2, 0.0], [0.2, 0.0]]),
        "realized_vol_21": np.array([[0.3, 0.1], [0.3, 0.1]]),
        "gics_onehot": np.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        ),  # (T, K, S=2)
    }
    out = holdings_exposures(weights, chars)
    # size: mean_t(sum_k w C) = mean(2.0, 3.0) = 2.5
    assert out["exposure_size"] == pytest.approx(2.5, **FLOAT_TOL)
    # value: mean(1.0, 0.5) = 0.75
    assert out["exposure_value"] == pytest.approx(0.75, **FLOAT_TOL)
    # sector HHI: t0 -> [1,0] HHI=1; t1 -> [0.5,0.5] HHI=0.5; mean=0.75
    assert out["sector_hhi"] == pytest.approx(0.75, **FLOAT_TOL)


def test_load_characteristic_panel_shape_and_finite(monkeypatch):
    from src.reporting import holdings_exposure as he

    dates = ["2020-01-02", "2020-01-03"]
    secids = ["101", "202"]

    def fake_equity(lake_root, dates, secids):
        t, k = len(dates), len(secids)
        return {
            "mktcap": np.full((t, k), 100.0),
            "ret_12_1": np.zeros((t, k)),
            "hv_21": np.full((t, k), 0.2),
        }

    def fake_bm_roe(lake_root, dates, secids):
        t, k = len(dates), len(secids)
        return {
            "bm": np.full((t, k), 0.5),
            "roe": np.full((t, k), 0.1),
        }

    def fake_gics(lake_root, secids):
        # 2 sectors
        return np.eye(len(secids), 2)

    monkeypatch.setattr(he, "_load_equity_chars", fake_equity)
    monkeypatch.setattr(he, "_load_fundamental_chars", fake_bm_roe)
    monkeypatch.setattr(he, "_load_gics_onehot", fake_gics)

    panels = he.load_characteristic_panel(dates, secids, lake_root="/tmp/fake_lake")
    assert panels["log_mktcap"].shape == (2, 2)
    assert panels["gics_onehot"].shape == (2, 2, 2)
    for key, arr in panels.items():
        assert np.isfinite(arr).all(), key
