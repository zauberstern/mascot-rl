"""Unit tests for Gatheral–Jacquier static-arbitrage screens."""
from __future__ import annotations

import pandas as pd
import pytest

from mascotrl.data.arbitrage_screens import (
    butterfly_violations,
    calendar_violations,
    filter_long_marks,
)
from mascotrl.data.duckdb_engine import OptionFilterConfig


def test_calendar_flags_decreasing_total_variance():
    # Same delta/cp: longer tenor must not have smaller w = iv^2 * tau.
    rows = []
    for days, iv in ((30, 0.20), (60, 0.10)):  # w_60 << w_30 → violation
        rows.append(
            {
                "secid": 1,
                "date": "2020-01-02",
                "days": days,
                "delta": 50,
                "cp_flag": "C",
                "impl_volatility": iv,
            }
        )
    viol = calendar_violations(pd.DataFrame(rows))
    assert len(viol) == 1
    assert int(viol.iloc[0]["n_violations"]) >= 1


def test_calendar_clean_when_w_nondecreasing():
    rows = []
    for days, iv in ((30, 0.20), (60, 0.20), (91, 0.20)):
        rows.append(
            {
                "secid": 1,
                "date": "2020-01-02",
                "days": days,
                "delta": 50,
                "cp_flag": "C",
                "impl_volatility": iv,
            }
        )
    viol = calendar_violations(pd.DataFrame(rows))
    assert viol.empty


def test_butterfly_flags_nonconvex_slice():
    # Middle call too expensive relative to wings → convexity fail.
    df = pd.DataFrame(
        {
            "secid": [1, 1, 1],
            "date": ["2020-01-02"] * 3,
            "exdate": ["2020-02-21"] * 3,
            "cp_flag": ["C"] * 3,
            "strike": [90.0, 100.0, 110.0],
            "mid": [12.0, 20.0, 2.0],  # middle wildly rich
        }
    )
    viol = butterfly_violations(df)
    assert len(viol) == 1


def test_butterfly_clean_on_linear_slice():
    # Linear in strike is weakly convex (butterfly ≈ 0); tolerate with eps.
    df = pd.DataFrame(
        {
            "secid": [1, 1, 1],
            "date": ["2020-01-02"] * 3,
            "exdate": ["2020-02-21"] * 3,
            "cp_flag": ["C"] * 3,
            "strike": [90.0, 100.0, 110.0],
            "mid": [15.0, 10.0, 5.0],
        }
    )
    viol = butterfly_violations(df)
    assert viol.empty


def test_filter_long_marks_drops_bad_keys():
    long = pd.DataFrame(
        {
            "secid": [1, 1, 2],
            "date": ["2020-01-02", "2020-01-03", "2020-01-02"],
            "mid": [1.0, 1.1, 2.0],
        }
    )
    out = filter_long_marks(long, {(1, "2020-01-02")})
    assert len(out) == 2
    assert set(zip(out["secid"].tolist(), out["date"].astype(str).tolist())) == {
        (1, "2020-01-03"),
        (2, "2020-01-02"),
    }


def test_option_filter_config_arb_flags_default_on():
    cfg = OptionFilterConfig()
    assert cfg.no_calendar_arbitrage is True
    assert cfg.no_butterfly_arbitrage is True
    assert cfg.drop_surface_arb_days is False
    disabled = OptionFilterConfig.disabled()
    assert disabled.no_calendar_arbitrage is False
    assert disabled.no_butterfly_arbitrage is False
