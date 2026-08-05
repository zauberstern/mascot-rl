"""TDD: spectrum multiple-testing helpers (Romano-Wolf, MDE, trial counts)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from mascotrl.eval.spectrum_multiple_testing import (
    n_trials_breakdown,
    paired_mde,
    romano_wolf_stepdown,
)
from scripts.build_spectrum_decision import beats_reference


def test_paired_mde_formula() -> None:
    assert paired_mde(1.0, 100) == pytest.approx(2.802 / 10.0)
    assert math.isnan(paired_mde(1.0, 0))


def test_n_trials_breakdown() -> None:
    d = n_trials_breakdown(20, 5, 1)
    assert d["n_trials"] == 100
    assert d["n_cells"] == 20
    assert d["n_seeds"] == 5
    assert d["n_cost_rungs"] == 1
    assert "formula" in d


def test_romano_wolf_stepdown_returns_adjusted_p() -> None:
    rng = np.random.default_rng(0)
    # Clear winner vs noise nulls.
    diffs = {
        "winner": list(0.5 + rng.normal(0, 0.05, size=40)),
        "null_a": list(rng.normal(0, 0.05, size=40)),
        "null_b": list(rng.normal(0, 0.05, size=40)),
    }
    out = romano_wolf_stepdown(diffs, n_boot=100, seed=0, alpha=0.05)
    assert out["protocol"] == "romano_wolf_stepdown_spectrum"
    assert "winner" in out["adjusted_pvalues"]
    assert out["adjusted_pvalues"]["winner"] <= out["adjusted_pvalues"]["null_a"]
    assert "winner" in out["rejected"] or out["adjusted_pvalues"]["winner"] < 0.1


def test_beats_reference_orientation() -> None:
    assert beats_reference(1.2, 1.0, orientation="higher_better") is True
    assert beats_reference(0.8, 1.0, orientation="higher_better") is False
    assert beats_reference(0.5, 0.8, orientation="lower_better") is True
    assert beats_reference(0.9, 0.8, orientation="lower_better") is False
    assert beats_reference(float("nan"), 1.0, orientation="higher_better") is False
