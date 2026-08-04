"""Exact-nonzero support Jaccard, exit, and reentry measures."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tests.conftest import FLOAT_TOL


def _fixture_path() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 0.5],
        ],
        dtype=np.float64,
    )


def test_support_measures_hand_computed() -> None:
    from src.reporting.behavior_metrics import (
        SUPPORT_EPS,
        measure_support_exit_rate,
        measure_support_jaccard_lag1,
        measure_support_reentry_rate,
        measure_support_size_mean,
    )

    assert SUPPORT_EPS == 1e-8
    w = _fixture_path()
    assert measure_support_size_mean(w) == pytest.approx(1.5, **FLOAT_TOL)
    assert measure_support_jaccard_lag1(w) == pytest.approx(1.0 / 3.0, **FLOAT_TOL)
    assert measure_support_exit_rate(w) == pytest.approx(1.0 / 3.0, **FLOAT_TOL)
    assert measure_support_reentry_rate(w) == pytest.approx(1.0 / 3.0, **FLOAT_TOL)


def test_equal_weight_path_full_support() -> None:
    from src.reporting.behavior_metrics import (
        measure_support_exit_rate,
        measure_support_jaccard_lag1,
        measure_support_reentry_rate,
        measure_support_size_mean,
    )

    k = 5
    w = np.full((8, k), 1.0 / k, dtype=np.float64)
    assert measure_support_size_mean(w) == pytest.approx(float(k), **FLOAT_TOL)
    assert measure_support_jaccard_lag1(w) == pytest.approx(1.0, **FLOAT_TOL)
    assert measure_support_exit_rate(w) == pytest.approx(0.0, **FLOAT_TOL)
    assert measure_support_reentry_rate(w) == pytest.approx(0.0, **FLOAT_TOL)


def test_single_row_nan_lag_measures() -> None:
    from src.reporting.behavior_metrics import (
        measure_support_exit_rate,
        measure_support_jaccard_lag1,
        measure_support_reentry_rate,
        measure_support_size_mean,
    )

    w = np.array([[0.5, 0.5, 0.0]], dtype=np.float64)
    assert measure_support_size_mean(w) == pytest.approx(2.0, **FLOAT_TOL)
    assert math.isnan(measure_support_jaccard_lag1(w))
    assert math.isnan(measure_support_exit_rate(w))
    assert math.isnan(measure_support_reentry_rate(w))
