"""TDD: equal_weight_collapse_guard detects a policy that never leaves 1/K."""
from __future__ import annotations

import numpy as np

from src.eval.collapse_guard import equal_weight_collapse_guard


def test_equal_weight_collapse_guard_fails_on_pure_ew_weights() -> None:
    k = 5
    weights = np.full((20, k), 1.0 / k)
    report = equal_weight_collapse_guard(weights)
    assert report["ok"] is False
    assert report["collapse_detected"] is True
    assert report["failures"]
    assert report["mean_l1_vs_ew"] < 0.05


def test_equal_weight_collapse_guard_passes_on_concentrated_weights() -> None:
    rng = np.random.default_rng(0)
    k = 5
    rows = []
    for _ in range(20):
        raw = rng.normal(size=k)
        w = np.exp(raw * 5.0)
        w = w / w.sum()
        rows.append(w)
    weights = np.stack(rows)
    report = equal_weight_collapse_guard(weights)
    assert report["ok"] is True
    assert report["collapse_detected"] is False
    assert report["mean_l1_vs_ew"] >= 0.05


def test_equal_weight_collapse_guard_returns_expected_keys() -> None:
    k = 4
    weights = np.full((10, k), 1.0 / k)
    report = equal_weight_collapse_guard(weights)
    for key in (
        "mean_hhi",
        "mean_l1_vs_ew",
        "mean_max_weight",
        "collapse_detected",
        "ok",
        "failures",
    ):
        assert key in report


def test_equal_weight_collapse_guard_custom_floor() -> None:
    k = 3
    weights = np.full((5, k), 1.0 / k)
    # A very low floor makes even EW pass.
    report = equal_weight_collapse_guard(weights, l1_vs_ew_floor=0.0)
    assert report["ok"] is True
