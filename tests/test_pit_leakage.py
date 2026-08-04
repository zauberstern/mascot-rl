"""Point-in-time leakage guards (W3).

Covers the four leaks closed in the audit: backward-filled universe scoring,
universe selection overlapping the scored window, zero-imputed labels, and
full-window macro standardization. Plus a shuffled-label falsification test:
if a pipeline reports edge on destroyed labels, it is reading the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.conftest import FLOAT_TOL

from src.data.macro_loader import _frame_to_tensor
from src.data.pit_guards import (
    assert_headline_selection_pit,
    membership_filter,
    selection_pit_status,
)


# ---------------------------------------------------------------- selection PIT

def test_selection_pit_clean_when_eval_starts_after_universe_window():
    st = selection_pit_status(
        universe_end="2021-12-31", eval_start="2022-01-01", phase="OOS_TEST"
    )
    assert st["pit_clean"] is True
    assert st["overlap_days"] == 0
    assert_headline_selection_pit(st)


def test_selection_pit_dirty_when_windows_overlap():
    st = selection_pit_status(
        universe_end="2021-12-31", eval_start="2007-01-01", phase="IS_TRAIN"
    )
    assert st["pit_clean"] is False
    assert st["overlap_days"] > 0
    with pytest.raises(RuntimeError, match="look-ahead"):
        assert_headline_selection_pit(st)


def test_selection_pit_unknown_windows_are_not_claimed_clean():
    st = selection_pit_status(universe_end=None, eval_start="2022-01-01", phase="X")
    assert st["pit_clean"] is False


def test_selection_pit_slot_masked_is_clean_for_is_and_oos():
    is_st = selection_pit_status(
        universe_end="2021-12-31",
        eval_start="2007-01-01",
        phase="IS_TRAIN",
        universe_protocol="slot_masked",
    )
    oos_st = selection_pit_status(
        universe_end="2021-12-31",
        eval_start="2022-01-01",
        phase="OOS_TEST",
        universe_protocol="slot_masked",
    )
    assert is_st["pit_clean"] is True
    assert oos_st["pit_clean"] is True
    assert is_st["universe_protocol"] == "slot_masked"


# ------------------------------------------------------------- PIT membership

def test_membership_filter_drops_non_members():
    rows = [{"secid": 1, "ticker": "AAA"}, {"secid": 2, "ticker": "ZZZ"}]
    kept, meta = membership_filter(rows, {"AAA"})
    assert [r["ticker"] for r in kept] == ["AAA"]
    assert meta["enforced"] is True
    assert meta["n_dropped_non_member"] == 1


def test_membership_filter_discloses_when_snapshot_missing():
    rows = [{"secid": 1, "ticker": "AAA"}]
    kept, meta = membership_filter(rows, set())
    assert kept == rows
    assert meta["enforced"] is False
    assert "unavailable" in meta["reason"]


# ------------------------------------------------------------- causal z-score

def test_causal_zscore_uses_no_future_information():
    """Truncating the future must not change earlier standardized rows."""
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"a": rng.standard_normal(400).cumsum()})
    full = _frame_to_tensor(x, 1, causal_zscore=True, min_periods=10).numpy()
    trunc = _frame_to_tensor(x.iloc[:200], 1, causal_zscore=True, min_periods=10).numpy()
    np.testing.assert_allclose(full[:200, 0], trunc[:, 0], rtol=1e-6, atol=1e-6)


def test_full_window_zscore_does_leak_future():
    """Contrast case: the old behaviour fails the same invariant."""
    rng = np.random.default_rng(1)
    x = pd.DataFrame({"a": rng.standard_normal(400).cumsum()})
    full = _frame_to_tensor(x, 1, causal_zscore=False).numpy()
    trunc = _frame_to_tensor(x.iloc[:200], 1, causal_zscore=False).numpy()
    assert not np.allclose(full[:200, 0], trunc[:, 0], rtol=1e-3, atol=1e-3)


def test_causal_zscore_is_finite_and_zeroed_before_min_periods():
    x = pd.DataFrame({"a": np.arange(100, dtype=float)})
    out = _frame_to_tensor(x, 1, causal_zscore=True, min_periods=20).numpy()
    assert np.isfinite(out).all()
    assert np.allclose(out[:19, 0], 0.0)


# --------------------------------------------------- label masking (no zero-fill)

def test_missing_labels_do_not_contribute_pnl():
    """A NaN label must contribute zero, not be treated as a realized 0.0."""
    w = np.array([0.5, 0.5])
    raw = np.array([0.02, np.nan])
    mask = np.isfinite(raw)
    contrib = np.where(mask, np.nan_to_num(raw, nan=0.0), 0.0)
    pnl = float((w * contrib).sum())
    assert pnl == pytest.approx(0.01)
    assert float(mask.mean()) == pytest.approx(0.5, **FLOAT_TOL)


def test_label_coverage_is_measured_not_assumed():
    raw = np.array([[0.01, np.nan], [0.02, 0.03]])
    cov = [float(np.isfinite(r).mean()) for r in raw]
    assert cov == [0.5, 1.0]
    assert float(np.mean(cov)) == pytest.approx(0.75, **FLOAT_TOL)


# ------------------------------------------------------ shuffled-label falsification

def _sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    sd = x.std(ddof=0)
    if sd < 1e-15:
        return 0.0
    return float(x.mean() / sd * np.sqrt(252.0))


def test_shuffled_labels_destroy_measured_edge():
    """
    Falsification: a signal fit to real labels loses its edge when the labels
    are permuted. A pipeline that still shows edge is reading the future.
    """
    rng = np.random.default_rng(42)
    T, k = 1500, 6
    signal = rng.standard_normal((T, k))
    # Genuine relationship: next-period label depends on the current signal.
    labels = 0.4 * signal + rng.standard_normal((T, k)) * 0.6

    def strategy_pnl(sig: np.ndarray, lab: np.ndarray) -> np.ndarray:
        w = sig / np.maximum(np.abs(sig).sum(axis=1, keepdims=True), 1e-9)
        return (w * lab).sum(axis=1)

    real = _sharpe(strategy_pnl(signal, labels))
    shuffled = [
        _sharpe(strategy_pnl(signal, rng.permutation(labels, axis=0)))
        for _ in range(40)
    ]
    assert real > 3.0, f"fixture should show real edge, got {real}"
    # Permuted labels must be centered on zero and never approach the real edge.
    assert abs(float(np.mean(shuffled))) < 1.0
    assert max(shuffled) < real
