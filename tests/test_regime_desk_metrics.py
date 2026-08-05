"""Tests for regime desk metrics helpers."""
from __future__ import annotations

import numpy as np
import pytest

from mascotrl.eval.regime_desk_metrics import (
    best_solo_expert,
    book_table_row,
    max_drawdown,
    per_regime_desk_stats,
    sharpe_annualized,
    weight_turnover_l1,
)
from mascotrl.eval.regime_desk_peers import causal_rolling_panel_returns, hrp_weights


def test_max_drawdown_and_turnover() -> None:
    wealth = np.array([1.0, 1.1, 1.05, 0.9, 1.0])
    assert max_drawdown(wealth) == pytest.approx((0.9 - 1.1) / 1.1)
    W = np.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    # |diff| L1: t1=1.0, t2=2.0; mean=1.5; half=0.75
    assert weight_turnover_l1(W) == pytest.approx(0.75)


def test_per_regime_desk_stats() -> None:
    rng = np.random.default_rng(0)
    t = 200
    turb = np.zeros(t, dtype=bool)
    turb[100:] = True
    books = {
        "fixed_share": rng.normal(0.001, 0.01, size=t),
        "equal_weight": rng.normal(0.0005, 0.01, size=t),
    }
    books["fixed_share"][turb] += 0.002
    out = per_regime_desk_stats(books, turb)
    assert out["turbulent"]["fixed_share"]["n_days"] == 100
    assert out["calm"]["equal_weight"]["n_days"] == 100
    assert np.isfinite(out["turbulent"]["fixed_share"]["sharpe"])


def test_best_solo_expert() -> None:
    R = np.array([[0.01, -0.02], [0.01, -0.01], [0.01, 0.0]], dtype=np.float64)
    L = -R
    solo = best_solo_expert(L, R, ["a", "b"])
    assert solo["name"] == "a"
    assert solo["index"] == 0
    row = book_table_row(R[:, 0], turnover=0.1)
    assert np.isfinite(row["sharpe"])
    assert row["turnover"] == 0.1


def test_hrp_weights_simplex() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.01, size=(80, 4))
    cov = np.cov(x, rowvar=False) + np.eye(4) * 1e-6
    w = hrp_weights(cov)
    assert w.shape == (4,)
    assert w.sum() == pytest.approx(1.0, abs=1e-8)
    assert np.all(w >= -1e-12)


def test_causal_rolling_hrp_and_eg_finite() -> None:
    rng = np.random.default_rng(2)
    t, k = 180, 5
    # Mild trend so peers have signal
    panel = rng.normal(0.0005, 0.01, size=(t, k))
    hrp = causal_rolling_panel_returns(panel, lookback=60, min_obs=30, mode="hrp")
    eg = causal_rolling_panel_returns(panel, lookback=60, min_obs=30, mode="eg")
    assert hrp["returns"] is not None
    assert eg["returns"] is not None
    assert np.isfinite(hrp["sharpe"]) or hrp["limitation"] is not None
    # Prefix stability: first 100 days of a longer run match a short run
    short = causal_rolling_panel_returns(panel[:100], lookback=60, min_obs=30, mode="hrp")
    long = causal_rolling_panel_returns(panel, lookback=60, min_obs=30, mode="hrp")
    a = short["returns"]
    b = long["returns"][:100]
    assert a is not None and b is not None
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() > 10:
        np.testing.assert_allclose(a[mask], b[mask], rtol=1e-10, atol=1e-12)
