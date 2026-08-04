"""P1: slot-aware surface-signal alignment for dynamic universe arms."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.surface_signals import (
    _canonical_secid_key,
    align_signals_to_panel,
    align_signals_to_slots,
)


def _toy_signals() -> pd.DataFrame:
    # Month-end publication dates; lag_days=1 makes them available next day.
    rows = [
        {"secid": 10, "date": "2014-01-31", "mfis_30": 0.11, "mfis_365": 0.21},
        {"secid": 20, "date": "2014-01-31", "mfis_30": 0.12, "mfis_365": 0.22},
        {"secid": 30, "date": "2014-01-31", "mfis_30": 0.13, "mfis_365": 0.23},
        {"secid": 10, "date": "2014-02-28", "mfis_30": 0.31, "mfis_365": 0.41},
        {"secid": 20, "date": "2014-02-28", "mfis_30": 0.32, "mfis_365": 0.42},
        {"secid": 30, "date": "2014-02-28", "mfis_30": 0.33, "mfis_365": 0.43},
    ]
    return pd.DataFrame(rows)


def test_align_signals_to_slots_follows_occupant_secid():
    signals = _toy_signals()
    dates = pd.date_range("2014-02-05", periods=3, freq="D")
    # Slot 0 holds secid 10 then rotates to 30; slot 1 holds 20 then None.
    slots_rows = [
        [10, 20],
        [10, 20],
        [30, None],
    ]
    out = align_signals_to_slots(
        signals,
        dates,
        slots_rows,
        lag_days=1,
        signal_names=["mfis_30", "mfis_365"],
    )
    assert set(out) == {"mfis_30", "mfis_365"}
    assert out["mfis_30"].shape == (3, 2)
    # Jan month-end (0.11/0.12) is available from Feb 1 onward under lag=1.
    np.testing.assert_allclose(out["mfis_30"][0, 0], 0.11)
    np.testing.assert_allclose(out["mfis_30"][0, 1], 0.12)
    np.testing.assert_allclose(out["mfis_30"][2, 0], 0.13)  # rotated to secid 30
    assert np.isnan(out["mfis_30"][2, 1])  # inactive slot


def test_align_signals_to_slots_matches_static_panel_when_slots_fixed():
    signals = _toy_signals()
    dates = pd.date_range("2014-02-05", periods=5, freq="D")
    secids = [10, 20, 30]
    slots_rows = [list(secids) for _ in range(len(dates))]
    panel = align_signals_to_panel(
        signals, dates, secids, lag_days=1, signal_names=["mfis_30"]
    )
    slotted = align_signals_to_slots(
        signals, dates, slots_rows, lag_days=1, signal_names=["mfis_30"]
    )
    np.testing.assert_allclose(panel["mfis_30"], slotted["mfis_30"], equal_nan=True)


def test_align_signals_to_slots_empty_signals_is_all_nan():
    dates = pd.date_range("2014-02-05", periods=2, freq="D")
    slots_rows = [[10, 20], [10, None]]
    out = align_signals_to_slots(
        pd.DataFrame(columns=["secid", "date", "mfis_30"]),
        dates,
        slots_rows,
        lag_days=1,
        signal_names=["mfis_30"],
    )
    assert out["mfis_30"].shape == (2, 2)
    assert np.all(np.isnan(out["mfis_30"]))


def test_float_and_int_secids_share_key_and_panel_column():
    assert _canonical_secid_key(101.0) == _canonical_secid_key(101)

    signals = pd.DataFrame(
        [{"secid": 101.0, "date": "2020-01-31", "mfis_30": 0.42}]
    )
    out = align_signals_to_panel(
        signals,
        pd.date_range("2020-02-02", periods=2, freq="D"),
        [101],
        lag_days=1,
        signal_names=["mfis_30"],
    )

    np.testing.assert_allclose(out["mfis_30"], [[0.42], [0.42]])
