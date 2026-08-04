"""PIT / look-ahead guards for surface signals (Phase E)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.surface_signals import (
    build_kelly_iv_images,
    cache_surface_signals,
    compute_surface_signals_panel,
    load_surface_signals,
    materialize_surface_signals_from_lake,
)


def _point(
    secid: int,
    date: str,
    days: int,
    delta: int,
    cp_flag: str,
    iv: float,
) -> dict:
    spot = 100.0
    d = abs(int(delta))
    if str(cp_flag).upper().startswith("P"):
        strike = spot * (1.0 - d / 200.0)
    else:
        strike = spot if int(delta) == 50 else spot * (1.0 + d / 200.0)
    return {
        "secid": secid,
        "date": pd.Timestamp(date),
        "days": int(days),
        "delta": int(delta),
        "cp_flag": cp_flag,
        "impl_volatility": float(iv),
        "impl_strike": float(strike),
        "impl_premium": max(0.5, abs(spot - strike) * 0.1 + 1.0),
        "dispersion": 0.01,
    }


def _two_month_panel() -> pd.DataFrame:
    """Two month-ends for one name; Jan ATM call=0.20, Feb ATM call=0.30."""
    rows = []
    for date, atm in (("2020-01-31", 0.20), ("2020-02-28", 0.30)):
        rows.append(_point(7, date, 30, 50, "C", atm))
        rows.append(_point(7, date, 30, -50, "P", atm + 0.02))
        rows.append(_point(7, date, 30, -20, "P", atm + 0.05))
        rows.append(_point(7, date, 365, 50, "C", atm + 0.03))
        # denser OTM strikes for svix / mf moments
        for d in (20, 25, 30, 40):
            rows.append(_point(7, date, 30, d, "C", atm - 0.01 + d * 0.0001))
            rows.append(_point(7, date, 30, -d, "P", atm + 0.04 + d * 0.0001))
            rows.append(_point(7, date, 60, d, "C", atm))
            rows.append(_point(7, date, 60, -d, "P", atm + 0.03))
            rows.append(_point(7, date, 91, d, "C", atm + 0.005))
            rows.append(_point(7, date, 91, -d, "P", atm + 0.035))
    return pd.DataFrame(rows)


def test_pit_future_mutation_does_not_change_past_signals():
    base = _two_month_panel()
    panel_full = compute_surface_signals_panel(base, month_end_only=True)
    t0 = pd.Timestamp("2020-01-31")
    past = panel_full.loc[panel_full["date"] == t0].reset_index(drop=True)
    assert len(past) == 1

    mutated = base.copy()
    # Corrupt all February IVs dramatically.
    mask = mutated["date"] > t0
    mutated.loc[mask, "impl_volatility"] = 9.99
    panel_mut = compute_surface_signals_panel(mutated, month_end_only=True)
    past_mut = panel_mut.loc[panel_mut["date"] == t0].reset_index(drop=True)

    signal_cols = [c for c in past.columns if c not in ("secid", "date")]
    for c in signal_cols:
        a = past.loc[0, c]
        b = past_mut.loc[0, c]
        if pd.isna(a) and pd.isna(b):
            continue
        assert np.isfinite(a) and np.isfinite(b), f"PIT breach on {c}: {a} vs {b}"
        assert float(a) == pytest.approx(float(b)), f"PIT breach on {c}: {a} vs {b}"

def test_d_iv_uses_only_prior_month_end():
    base = _two_month_panel()
    panel = compute_surface_signals_panel(base, month_end_only=True)
    jan = panel.loc[panel["date"] == pd.Timestamp("2020-01-31")].iloc[0]
    feb = panel.loc[panel["date"] == pd.Timestamp("2020-02-28")].iloc[0]
    assert np.isnan(jan["d_iv_call_1m"])
    assert feb["d_iv_call_1m"] == pytest.approx(0.30 - 0.20)


def test_kelly_month_end_filter_ignores_intramonth_future():
    rows = []
    # Month-end Jan
    rows.append(_point(1, "2020-01-31", 30, 50, "C", 0.20))
    # Mid-Feb should not pollute Jan Kelly slice when dates filtered to Jan month-end
    rows.append(_point(1, "2020-02-15", 30, 50, "C", 0.99))
    rows.append(_point(1, "2020-02-28", 30, 50, "C", 0.30))
    df = pd.DataFrame(rows)
    dates = [pd.Timestamp("2020-01-31")]
    cube = build_kelly_iv_images(df, secids=[1], dates=dates)
    assert cube.shape == (1, 1, 11, 34)
    # ATM call 30d is at tenor index for 30, call delta 50 among the 34 slots
    # puts (-90..-10) then calls (10..90); call 50 is put17 + index of 50 in calls.
    tenors = (10, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730)
    deltas_put = tuple(range(-90, -5, 5))
    deltas_call = tuple(range(10, 95, 5))
    ti = tenors.index(30)
    di = len(deltas_put) + deltas_call.index(50)
    assert cube[0, 0, ti, di] == pytest.approx(0.20)


def test_cache_roundtrip(tmp_path):
    base = _two_month_panel()
    panel = compute_surface_signals_panel(base, month_end_only=True)
    path = tmp_path / "sig.parquet"
    cache_surface_signals(panel, path)
    loaded = load_surface_signals(path)
    pd.testing.assert_frame_equal(
        panel.reset_index(drop=True),
        loaded.reset_index(drop=True),
        check_dtype=False,
    )


def test_materialize_missing_lake_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        materialize_surface_signals_from_lake(
            tmp_path / "no_lake",
            secids=[1],
            start="2020-01-01",
            end="2020-12-31",
        )
