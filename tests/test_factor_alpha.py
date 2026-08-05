"""Characteristic sorts and factor-adjusted alpha (W7).

The threat being tested for: Goyal and Saretto (2024, RFS 38(6)) and the Dallas
Fed IPCA study (WP 2214) find factor models absorb apparent equity-option alpha.
These tests lock the machinery that would detect that.
"""
from __future__ import annotations

import numpy as np
import pytest

from mascotrl.eval.factor_alpha import (
    FEASIBLE_CHARACTERISTICS,
    UNAVAILABLE_CHARACTERISTICS,
    attach_factor_alpha,
    build_characteristics,
    build_option_factors,
    factor_alpha,
    long_short_sort,
)


# ------------------------------------------------------------------------ sorts

def test_long_short_sort_is_dollar_neutral_and_lagged():
    rng = np.random.default_rng(0)
    T, K = 200, 20
    char = rng.standard_normal((T, K))
    # Label depends on the PREVIOUS day's characteristic, so a lagged sort earns.
    labels = np.zeros((T, K))
    labels[:] = 0.01 * char + rng.standard_normal((T, K)) * 0.002
    out = long_short_sort(char, labels)
    assert out["construction"] == "equal_weighted_quintile_long_short_lagged_sort"
    assert out["n_days"] > 150
    assert out["mean"] > 0  # positive relationship should be picked up


def test_long_short_sort_earns_nothing_on_unrelated_characteristic():
    rng = np.random.default_rng(1)
    T, K = 400, 20
    char = rng.standard_normal((T, K))
    labels = rng.standard_normal((T, K)) * 0.01
    out = long_short_sort(char, labels)
    assert abs(out["mean"]) < 0.002


def test_long_short_sort_skips_thin_cross_sections():
    char = np.full((50, 4), np.nan)
    labels = np.zeros((50, 4))
    out = long_short_sort(char, labels)
    assert out["n_days"] == 0 or np.isnan(out["mean"])


def test_long_short_sort_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        long_short_sort(np.zeros((10, 3)), np.zeros((10, 4)))


def test_missing_labels_do_not_leak_into_sort_returns():
    char = np.tile(np.arange(10.0), (30, 1))
    labels = np.full((30, 10), np.nan)
    out = long_short_sort(char, labels)
    assert all(not np.isfinite(x) or x == 0.0 for x in out["pnl"])


# -------------------------------------------------------------- characteristics

def test_build_characteristics_produces_feasible_set():
    T, K = 120, 8
    rng = np.random.default_rng(2)
    iv = 0.25 + rng.standard_normal((T, K)) * 0.02
    ba = np.full((T, K), 0.05)
    mid = np.full((T, K), 2.0)
    hv = 0.28 + rng.standard_normal((T, K)) * 0.02
    ivol = np.abs(rng.standard_normal((T, K))) * 0.3
    chars = build_characteristics(
        atm_iv=iv, bid_ask_spread=ba, mid=mid, realized_vol=hv, idio_vol=ivol
    )
    for name in FEASIBLE_CHARACTERISTICS:
        assert name in chars, name
        assert chars[name].shape == (T, K)


def test_vol_deviation_is_log_hv_over_iv():
    iv = np.full((10, 2), 0.20)
    hv = np.full((10, 2), 0.40)
    chars = build_characteristics(
        atm_iv=iv,
        bid_ask_spread=np.full((10, 2), 0.01),
        mid=np.full((10, 2), 1.0),
        realized_vol=hv,
    )
    assert chars["vol_deviation"][0, 0] == pytest.approx(np.log(2.0))


def test_option_spread_is_relative_to_mid():
    chars = build_characteristics(
        atm_iv=np.full((5, 1), 0.2),
        bid_ask_spread=np.full((5, 1), 0.10),
        mid=np.full((5, 1), 2.0),
    )
    assert chars["option_spread"][0, 0] == pytest.approx(0.05)


def test_vol_of_vol_is_causal():
    """Vol-of-vol at t must use only IV strictly before t."""
    T, K = 100, 1
    iv = np.zeros((T, K))
    iv[:50] = 0.2          # calm first half
    iv[50:] = 0.2 + np.arange(50).reshape(-1, 1) * 0.01  # then trending
    chars = build_characteristics(
        atm_iv=iv,
        bid_ask_spread=np.full((T, K), 0.01),
        mid=np.full((T, K), 1.0),
        vol_of_vol_window=20,
    )
    vov = chars["vol_of_vol"]
    # Row 40 sees only the calm window -> near zero dispersion.
    assert vov[40, 0] == pytest.approx(0.0, abs=1e-9)
    # Later rows see the trend.
    assert vov[90, 0] > 0.01


# ------------------------------------------------------------------- factor set

def test_option_factors_include_market_leg_and_document_omissions():
    rng = np.random.default_rng(3)
    T, K = 200, 10
    labels = rng.standard_normal((T, K)) * 0.01
    chars = build_characteristics(
        atm_iv=0.2 + rng.standard_normal((T, K)) * 0.01,
        bid_ask_spread=np.full((T, K), 0.02),
        mid=np.full((T, K), 2.0),
        realized_vol=0.22 + rng.standard_normal((T, K)) * 0.01,
    )
    fac = build_option_factors(labels, chars)
    assert "option_market_ew" in fac["factors"]
    assert "hv_minus_iv" in fac["factors"]
    assert fac["model"] == "HVX_proxy"
    # Omissions must be stated, since missing factors bias alpha upward.
    assert "cash_holdings" in fac["omitted_factors"]
    assert set(UNAVAILABLE_CHARACTERISTICS) <= set(fac["omitted_factors"])
    assert "Goyal and Saretto" in fac["caveat"]


# ---------------------------------------------------------------------- alpha

def test_alpha_is_zero_when_strategy_is_a_factor_combination():
    """A strategy that is a pure factor bet must have no alpha."""
    rng = np.random.default_rng(4)
    n = 800
    f1 = rng.standard_normal(n) * 0.01
    f2 = rng.standard_normal(n) * 0.01
    strat = 0.7 * f1 + 0.3 * f2
    out = factor_alpha(strat, {"f1": f1, "f2": f2})
    assert out["ok"] is True
    assert out["alpha_daily"] == pytest.approx(0.0, abs=1e-12)
    assert out["betas"]["f1"] == pytest.approx(0.7, rel=1e-6)
    assert out["r_squared"] > 0.999
    assert out["alpha_significant_05"] is False


def test_alpha_detected_when_strategy_has_genuine_intercept():
    rng = np.random.default_rng(5)
    n = 1200
    f1 = rng.standard_normal(n) * 0.01
    strat = 0.0006 + 0.5 * f1 + rng.standard_normal(n) * 0.001
    out = factor_alpha(strat, {"f1": f1})
    assert out["alpha_daily"] == pytest.approx(0.0006, abs=2e-4)
    assert out["alpha_t_hac"] > 2.0
    assert out["alpha_significant_05"] is True


def test_factor_model_absorbs_alpha_that_raw_mean_would_show():
    """The Goyal-Saretto threat: raw edge vanishes once factors are included."""
    rng = np.random.default_rng(6)
    n = 1000
    f1 = 0.0008 + rng.standard_normal(n) * 0.01  # factor with a premium
    strat = 1.0 * f1                              # pure exposure, no skill
    raw_mean = float(np.mean(strat))
    out = factor_alpha(strat, {"f1": f1})
    assert raw_mean > 0                            # looks profitable raw
    assert out["alpha_daily"] == pytest.approx(0.0, abs=1e-12)
    assert out["alpha_significant_05"] is False


def test_alpha_uses_hac_and_reports_lags():
    rng = np.random.default_rng(7)
    n = 900
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + e[t]
    out = factor_alpha(0.001 + x * 0.001, {"f": rng.standard_normal(n) * 0.01})
    assert out["hac_lags"] > 0
    assert abs(out["alpha_t_hac"]) < abs(out["alpha_t_iid"])


def test_alpha_requires_minimum_sample():
    out = factor_alpha([0.01] * 10, {"f": [0.0] * 10})
    assert out["ok"] is False
    assert "30" in out["reason"]


def test_attach_writes_report_section():
    rng = np.random.default_rng(8)
    T, K = 300, 10
    labels = rng.standard_normal((T, K)) * 0.01
    chars = build_characteristics(
        atm_iv=0.2 + rng.standard_normal((T, K)) * 0.01,
        bid_ask_spread=np.full((T, K), 0.02),
        mid=np.full((T, K), 2.0),
        realized_vol=0.22 + rng.standard_normal((T, K)) * 0.01,
    )
    report: dict = {}
    out = attach_factor_alpha(
        report,
        strategy_pnl=(rng.standard_normal(T - 1) * 0.01).tolist(),
        labels=labels,
        characteristics=chars,
    )
    assert report["factor_alpha"] is out
    assert "alpha" in out and "characteristic_sorts" in out
    assert "vol_deviation" in out["characteristic_sorts"]
