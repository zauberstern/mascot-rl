"""C7: gate1/gate2/gate3 -- one shared module for every promotion gate."""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.spectrum_gates import compute_gate1, compute_gate2, compute_gate3
from tests.conftest import FLOAT_TOL


def test_gate1_passes_above_threshold_fails_below() -> None:
    ok = compute_gate1({"break_even_spread_multiplier": 0.5, "cost_source": "om_touch"})
    assert ok["pass"] is True
    assert ok["decision"] == "continue_positive_framing"

    bad = compute_gate1({"break_even_spread_multiplier": 0.1, "cost_source": "om_touch"})
    assert bad["pass"] is False
    assert bad["decision"] == "pivot_negative_economic_framing"


def test_gate1_handles_missing_or_nan_break_even() -> None:
    out = compute_gate1({"break_even_spread_multiplier": float("nan")})
    assert out["pass"] is False
    out2 = compute_gate1({})
    assert out2["pass"] is False


def test_gate2_recovers_known_positive_alpha_with_significant_t() -> None:
    rng = np.random.default_rng(0)
    n = 400
    factors = rng.normal(0.0, 0.01, size=(n, 4))
    true_alpha = 0.004
    betas = np.array([1.0, 0.3, -0.2, 0.1])
    noise = rng.normal(0.0, 0.002, size=n)
    y = true_alpha + factors @ betas + noise
    out = compute_gate2(y, factors)
    assert out["positive_edge"] is True
    assert out["alpha"] > 0
    assert out["t_stat"] > 2.0


def test_gate2_zero_alpha_no_edge() -> None:
    rng = np.random.default_rng(1)
    n = 300
    factors = rng.normal(0.0, 0.01, size=(n, 4))
    betas = np.array([1.0, 0.0, 0.0, 0.0])
    y = factors @ betas + rng.normal(0.0, 0.02, size=n)
    out = compute_gate2(y, factors)
    assert out["positive_edge"] is False


def test_gate3_passes_when_beats_best_baseline() -> None:
    out = compute_gate3(0.9, {"equal_weight": 0.4, "olps_ons": 0.6, "kelly_cnn": 0.5})
    assert out["pass"] is True
    assert out["best_baseline"] == "olps_ons"
    assert out["edge_vs_best_baseline"] == pytest.approx(0.9 - 0.6, **FLOAT_TOL)
    assert out["n_beaten"] == 3


def test_gate3_fails_when_best_baseline_beats_policy() -> None:
    out = compute_gate3(0.5, {"equal_weight": 0.4, "olps_ons": 0.6})
    assert out["pass"] is False
    assert out["beats"] == {"equal_weight": True, "olps_ons": False}


def test_gate3_require_beat_all_stricter_than_beat_best() -> None:
    partial = compute_gate3(0.55, {"equal_weight": 0.4, "olps_ons": 0.6}, require_beat_all=False)
    strict = compute_gate3(0.55, {"equal_weight": 0.4, "olps_ons": 0.6}, require_beat_all=True)
    assert partial["pass"] is False  # does not beat olps_ons (best)
    assert strict["pass"] is False

    beats_all = compute_gate3(0.7, {"equal_weight": 0.4, "olps_ons": 0.6}, require_beat_all=True)
    assert beats_all["pass"] is True


def test_gate3_empty_baselines_never_passes() -> None:
    out = compute_gate3(1.0, {})
    assert out["pass"] is False
    assert out["best_baseline"] is None


def test_gate3_ignores_nan_baseline_entries() -> None:
    out = compute_gate3(0.5, {"good": 0.3, "broken": float("nan")})
    assert "broken" not in out["baselines"]
    assert out["n_baselines"] == 1
