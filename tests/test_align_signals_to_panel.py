"""B1: align_signals_to_panel forward-fills month-end signals to daily with
a publication lag, and is PIT-safe (no value at date t derives from a
month-end after t - lag_days)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mascotrl.data.surface_signals import align_signals_to_panel


def _signals_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "secid": [7, 7, 8, 8],
            "date": pd.to_datetime(
                ["2020-01-31", "2020-02-28", "2020-01-31", "2020-02-28"]
            ),
            "iv_skew_30d": [0.10, 0.20, -0.05, -0.15],
        }
    )


def test_align_shapes_match_dates_and_secids() -> None:
    signals = _signals_df()
    dates = pd.bdate_range("2020-01-01", "2020-03-31")
    out = align_signals_to_panel(signals, dates, [7, 8], lag_days=1, signal_names=["iv_skew_30d"])
    assert set(out.keys()) == {"iv_skew_30d"}
    assert out["iv_skew_30d"].shape == (len(dates), 2)


def test_publication_lag_delays_visibility() -> None:
    signals = _signals_df()
    dates = pd.bdate_range("2020-01-30", "2020-02-05")
    out = align_signals_to_panel(signals, dates, [7], lag_days=1, signal_names=["iv_skew_30d"])
    col = out["iv_skew_30d"][:, 0]
    idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    # Jan-31 value not visible on Jan-31 itself (lag=1) ...
    assert np.isnan(col[idx[pd.Timestamp("2020-01-31")]])
    # ... but visible starting Feb-1 (the next requested date once lag has
    # elapsed) and holds until the next release.
    first_visible = next(i for i, v in enumerate(col) if np.isfinite(v))
    assert dates[first_visible] >= pd.Timestamp("2020-02-01")
    assert col[first_visible] == pytest.approx(0.10)


def test_no_lag_makes_month_end_value_same_day_visible() -> None:
    signals = _signals_df()
    dates = pd.bdate_range("2020-01-30", "2020-02-03")
    out = align_signals_to_panel(signals, dates, [7], lag_days=0, signal_names=["iv_skew_30d"])
    col = out["iv_skew_30d"][:, 0]
    idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    assert col[idx[pd.Timestamp("2020-01-31")]] == pytest.approx(0.10)


def test_pit_future_release_never_leaks_into_past_dates() -> None:
    """Mutating the Feb-28 (future) release must not change any value at
    or before Jan-31 + lag."""
    signals = _signals_df()
    dates = pd.bdate_range("2020-01-01", "2020-02-15")
    base = align_signals_to_panel(signals, dates, [7, 8], lag_days=1, signal_names=["iv_skew_30d"])

    mutated_df = signals.copy()
    mutated_df.loc[mutated_df["date"] == pd.Timestamp("2020-02-28"), "iv_skew_30d"] = 999.0
    mutated = align_signals_to_panel(
        mutated_df, dates, [7, 8], lag_days=1, signal_names=["iv_skew_30d"]
    )
    assert np.allclose(
        base["iv_skew_30d"], mutated["iv_skew_30d"], equal_nan=True
    ), "future release leaked into dates before its own publication"


def test_missing_secid_column_is_all_nan() -> None:
    signals = _signals_df()
    dates = pd.bdate_range("2020-01-01", "2020-03-01")
    out = align_signals_to_panel(
        signals, dates, [7, 999], lag_days=1, signal_names=["iv_skew_30d"]
    )
    assert np.isnan(out["iv_skew_30d"][:, 1]).all()


def test_empty_signals_returns_all_nan_for_requested_names() -> None:
    dates = pd.bdate_range("2020-01-01", "2020-01-10")
    out = align_signals_to_panel(
        pd.DataFrame(columns=["secid", "date"]), dates, [1, 2], signal_names=["iv_skew_30d", "mfiv_30"]
    )
    assert set(out.keys()) == {"iv_skew_30d", "mfiv_30"}
    for arr in out.values():
        assert arr.shape == (len(dates), 2)
        assert np.isnan(arr).all()
