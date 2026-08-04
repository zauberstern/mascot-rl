"""WP-S6: spectrum campaign trains all configured seeds."""
from __future__ import annotations

import pytest
from tests.conftest import FLOAT_TOL

import numpy as np

from scripts.run_spectrum_campaign import _aggregate_spectrum_seed_arts


def test_aggregate_three_seeds() -> None:
    arts = [
        {"seed": 0, "sharpe": 0.5, "sharpe_mean": 0.5},
        {"seed": 1, "sharpe": 0.7, "sharpe_mean": 0.7},
        {"seed": 2, "sharpe": 0.6, "sharpe_mean": 0.6},
    ]
    out = _aggregate_spectrum_seed_arts(arts)
    assert out["n_seeds"] == 3
    assert len(out["seed_results"]) == 3
    assert out["sharpe_mean"] == np.mean([0.5, 0.7, 0.6])
    assert "sharpe_std" in out


def test_aggregate_single_seed_preserves() -> None:
    arts = [{"seed": 0, "sharpe": 1.2, "other": "x"}]
    out = _aggregate_spectrum_seed_arts(arts)
    assert out["n_seeds"] == 1
    assert out["sharpe_mean"] == pytest.approx(1.2, **FLOAT_TOL)
    assert out["other"] == "x"
