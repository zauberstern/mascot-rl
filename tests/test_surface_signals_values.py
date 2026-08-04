"""Value-reproduction tests for surface signal constructions (Phase E)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.surface_signals import (
    SURFACE_SIGNAL_NAMES,
    build_kelly_iv_images,
    compute_surface_signals_panel,
    extract_grid_point,
)


def _row(
    secid: int,
    date: str,
    days: int,
    delta: int,
    cp_flag: str,
    iv: float,
    *,
    strike: float | None = None,
    premium: float | None = None,
    dispersion: float = 0.01,
) -> dict:
    spot = 100.0
    d = int(delta)
    if strike is None:
        ad = abs(d)
        if str(cp_flag).upper().startswith("P"):
            strike = spot * (1.0 - ad / 200.0)
        elif d == 50:
            strike = spot
        else:
            strike = spot * (1.0 + ad / 200.0)
    if premium is None:
        premium = max(0.1, 0.2 * abs(spot - strike) + 1.0)
    return {
        "secid": secid,
        "date": pd.Timestamp(date),
        "days": days,
        "delta": d,
        "cp_flag": cp_flag,
        "impl_volatility": iv,
        "impl_strike": float(strike),
        "impl_premium": float(premium),
        "dispersion": float(dispersion),
    }


def _mini_surface_known_ivs() -> pd.DataFrame:
    """Single (secid, date) with exact IVs for skew / slope / convexity / CW."""
    date = "2020-01-31"
    sid = 101
    rows = [
        # 30d smirk + ATM
        _row(sid, date, 30, -20, "P", 0.30),
        _row(sid, date, 30, 50, "C", 0.20),
        _row(sid, date, 30, -50, "P", 0.25),
        # term slope ATM
        _row(sid, date, 365, 50, "C", 0.25),
        # convexity wings
        _row(sid, date, 30, 25, "C", 0.22),
        _row(sid, date, 30, 75, "C", 0.24),
        # CW matched |delta| pairs at 30/60/91
        _row(sid, date, 30, 20, "C", 0.21),
        _row(sid, date, 30, -25, "P", 0.28),
        _row(sid, date, 30, 30, "C", 0.205),
        _row(sid, date, 30, -30, "P", 0.27),
        _row(sid, date, 30, 40, "C", 0.202),
        _row(sid, date, 30, -40, "P", 0.26),
        _row(sid, date, 60, 20, "C", 0.215),
        _row(sid, date, 60, -20, "P", 0.29),
        _row(sid, date, 60, 50, "C", 0.21),
        _row(sid, date, 60, -50, "P", 0.255),
        _row(sid, date, 91, 20, "C", 0.22),
        _row(sid, date, 91, -20, "P", 0.285),
        _row(sid, date, 91, 50, "C", 0.215),
        _row(sid, date, 91, -50, "P", 0.26),
    ]
    return pd.DataFrame(rows)

def test_surface_signal_names_nonempty():
    assert isinstance(SURFACE_SIGNAL_NAMES, tuple)
    assert "iv_skew_30d" in SURFACE_SIGNAL_NAMES
    assert "mw_xs" in SURFACE_SIGNAL_NAMES
    assert "surface_quality" in SURFACE_SIGNAL_NAMES


def test_extract_grid_point_exact():
    df = _mini_surface_known_ivs()
    g = df  # single group
    assert extract_grid_point(g, days=30, delta=-20, cp_flag="P") == pytest.approx(0.30)
    assert extract_grid_point(g, days=30, delta=50, cp_flag="C") == pytest.approx(0.20)
    assert np.isnan(extract_grid_point(g, days=10, delta=50, cp_flag="C"))


def test_exact_skew_slope_convexity_cw():
    df = _mini_surface_known_ivs()
    panel = compute_surface_signals_panel(df, month_end_only=False)
    assert len(panel) == 1
    row = panel.iloc[0]
    assert row["iv_skew_30d"] == pytest.approx(0.30 - 0.20)
    assert row["iv_term_slope"] == pytest.approx(0.25 - 0.20)
    assert row["iv_convexity_30d"] == pytest.approx(0.22 + 0.24 - 2 * 0.20)
    # CW: equal-weight mean of available matched pairs
    pairs = [
        0.21 - 0.30,  # 30d |20|
        0.22 - 0.28,  # 30d |25|
        0.205 - 0.27,  # 30d |30|
        0.202 - 0.26,  # 30d |40|
        0.20 - 0.25,  # 30d |50|
        0.215 - 0.29,  # 60d |20|
        0.21 - 0.255,  # 60d |50|
        0.22 - 0.285,  # 91d |20|
        0.215 - 0.26,  # 91d |50|
    ]
    assert row["cw_vol_spread"] == pytest.approx(float(np.mean(pairs)))


def test_median_smirk_positive_on_steep_panel():
    """Xing-Zhang-Zhao: steep smirks → positive IV(P,-20)-IV(C,50); median ~0.05."""
    rows = []
    for i, sid in enumerate(range(1, 21)):
        date = "2020-06-30"
        smirk = 0.05 + 0.001 * (i - 10)  # centered near 0.05
        atm = 0.20
        rows.append(_row(sid, date, 30, -20, "P", atm + smirk))
        rows.append(_row(sid, date, 30, 50, "C", atm))
        rows.append(_row(sid, date, 365, 50, "C", atm + 0.02))
    df = pd.DataFrame(rows)
    panel = compute_surface_signals_panel(df, month_end_only=False)
    med = float(np.nanmedian(panel["iv_skew_30d"].to_numpy()))
    assert med == pytest.approx(0.05, abs=0.01)


def test_surface_quality_in_unit_interval():
    df = _mini_surface_known_ivs()
    # Pad to a partial grid so quality is in (0, 1)
    panel = compute_surface_signals_panel(df, month_end_only=False)
    q = float(panel.iloc[0]["surface_quality"])
    assert 0.0 <= q <= 1.0
    assert q < 1.0  # far fewer than 374 points


def test_vmp_os_ratio_borrow_from_indexed_aux_tables():
    """B1: aux tables (hv, option/equity volume, borrow) feed vmp / os_ratio
    / borrow_rate through the pre-indexed lookup."""
    df = _mini_surface_known_ivs()
    secid = int(df["secid"].iloc[0])
    date = pd.Timestamp(df["date"].iloc[0])
    hv = pd.DataFrame({"secid": [secid], "date": [date], "hv": [0.10]})
    option_volume = pd.DataFrame({"secid": [secid], "date": [date], "option_volume": [500.0]})
    equity_volume = pd.DataFrame({"secid": [secid], "date": [date], "equity_volume": [10000.0]})
    borrow = pd.DataFrame({"secid": [secid], "date": [date], "borrow_rate": [0.02]})
    panel = compute_surface_signals_panel(
        df,
        hv=hv,
        option_volume=option_volume,
        equity_volume=equity_volume,
        borrow=borrow,
        month_end_only=False,
    )
    row = panel.iloc[0]
    assert row["borrow_rate"] == pytest.approx(0.02)
    assert row["os_ratio"] == pytest.approx(500.0 / 10000.0)
    assert np.isfinite(row["vmp"])


def test_aux_lookup_ignores_non_matching_secid_or_date():
    df = _mini_surface_known_ivs()
    other_secid = int(df["secid"].iloc[0]) + 1
    borrow = pd.DataFrame(
        {"secid": [other_secid], "date": [pd.Timestamp(df["date"].iloc[0])], "borrow_rate": [0.02]}
    )
    panel = compute_surface_signals_panel(df, borrow=borrow, month_end_only=False)
    assert np.isnan(panel.iloc[0]["borrow_rate"])


def test_kelly_image_shape():
    df = _mini_surface_known_ivs()
    dates = [pd.Timestamp("2020-01-31")]
    secids = [101]
    cube = build_kelly_iv_images(df, secids=secids, dates=dates)
    assert cube.shape == (1, 1, 11, 34)
    assert np.isnan(cube).any() or np.isfinite(cube).any()
