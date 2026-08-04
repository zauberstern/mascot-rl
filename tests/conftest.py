"""Shared pytest fixtures and import-path bootstrap."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

FLOAT_TOL = dict(abs=1e-6, rel=1e-5)
"""Default tolerance dict for pytest.approx on financial floats."""


@pytest.fixture
def rng():
    """Deterministic RNG for exact-reproducibility unit tests."""
    return np.random.default_rng(42)


@pytest.fixture(params=range(5))
def rng_multi(request):
    """Multi-seed RNG for stochastic property tests (5 independent seeds)."""
    return np.random.default_rng(request.param)


@pytest.fixture
def torch_deterministic():
    """Lock torch + CUDA RNG for reproducibility tests."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True, warn_only=True)
    yield
    torch.use_deterministic_algorithms(False)


def passing_gate_bundle() -> dict:
    """Synthetic bundle that clears every locked G0-G6 predicate."""
    return {
        "g0": {
            "explicit_arm": True,
            "pit": True,
            "friction_parity": True,
            "no_silent_delist": True,
            "no_rbergomi_ancestry": True,
            "eq_allowlist": True,
        },
        "g1": {
            "residual_sr": 0.15,
            "months_positive": 8,
            "n_months": 12,
        },
        "g2": {
            "median_residual_sr": 0.30,
            "p10_residual_sr": 0.05,
            "n_seeds": 10,
        },
        "g3": {
            "p05_sr": 0.10,
            "median_sr": 0.55,
            "ensemble_residual_return": 0.02,
            "bootstrap_ci_low_median": 0.01,
            "min_alpha_annual": 0.0,
            "max_seed_profit_share": 0.40,
            "n_positive_seeds": 18,
            "n_seeds": 30,
        },
        "g4": {
            "path_srs": [0.6, 0.55, 0.7, 0.52, 0.8],
            "n_positive_combos": 13,
            "n_combos": 15,
        },
        "g5": {
            "hac_t": 3.2,
            "dsr": 0.97,
            "pbo": 0.08,
            "spa_p": 0.03,
        },
        "g6": {
            "residual_sr_1_5x": 0.30,
            "alpha_1_5x": 0.01,
            "n_paths_positive_1_5x": 4,
            "n_paths": 5,
            "capacity_alpha_10m": 0.005,
            "published_1x": True,
            "published_1_5x": True,
            "published_3x": True,
        },
    }


def passing_collapse_guard() -> dict:
    return {"ok": True, "collapse_detected": False, "collapse_failures": []}


def capital_gate_pass_extras() -> dict:
    """Bundle + collapse guard for protocol hygiene fixtures."""
    return {
        "bundle": passing_gate_bundle(),
        "collapse_guard": passing_collapse_guard(),
    }
