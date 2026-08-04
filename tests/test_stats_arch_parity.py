"""Strict parity: hand-rolled stationary bootstrap vs arch."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("arch")

from src.eval.arch_bootstrap import (
    block_bootstrap_metric_ci_arch,
    stationary_bootstrap_indices_arch,
)
from src.eval.stats_rigor import block_bootstrap_metric_ci, stationary_bootstrap_indices


def test_arch_indices_length_and_range():
    idx = stationary_bootstrap_indices_arch(50, block_mean=5, seed=1)
    assert idx.shape == (50,)
    assert idx.min() >= 0 and idx.max() < 50


def test_custom_indices_length_and_range():
    rng = np.random.default_rng(1)
    idx = stationary_bootstrap_indices(50, block_mean=5, rng=rng)
    assert idx.shape == (50,)
    assert idx.min() >= 0 and idx.max() < 50


def test_bootstrap_ci_distributional_parity():
    """Point estimate identical; bootstrap moments close under many reps."""
    rng = np.random.default_rng(42)
    r = rng.normal(0.0005, 0.01, size=200)
    custom = block_bootstrap_metric_ci(
        r, metric="sharpe", n_boot=200, block_mean=5, seed=7, periods=252
    )
    arch_ci = block_bootstrap_metric_ci_arch(
        r, metric="sharpe", n_boot=200, block_mean=5, seed=7, periods=252
    )
    assert custom["point"] == pytest.approx(arch_ci["point"], rel=0, abs=1e-12)
    # Distributional parity (RNG streams differ): means within a band of each other
    assert abs(custom["boot_mean"] - arch_ci["boot_mean"]) < 0.35
    assert abs(custom["boot_std"] - arch_ci["boot_std"]) < 0.35
    assert custom["ci_low"] < custom["point"] < custom["ci_high"]
    assert arch_ci["ci_low"] < arch_ci["point"] < arch_ci["ci_high"]


def test_constant_returns_sharpe_nan_or_zero_vol_guard():
    r = np.zeros(30)
    custom = block_bootstrap_metric_ci(r, n_boot=20, seed=0)
    # Finite or nan both acceptable; must not raise
    assert "point" in custom


def test_backend_stamp_custom_and_arch():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, size=80)
    custom = block_bootstrap_metric_ci(r, n_boot=50, seed=1, backend="custom")
    assert custom["backend"] == "custom"
    assert np.isfinite(custom["ci_low"]) and np.isfinite(custom["ci_high"])
    arch_ci = block_bootstrap_metric_ci(r, n_boot=50, seed=1, backend="arch")
    assert arch_ci["backend"] == "arch"
    assert np.isfinite(arch_ci["ci_low"]) and np.isfinite(arch_ci["ci_high"])


def test_resolve_bootstrap_backend():
    from src.eval.arch_bootstrap import resolve_bootstrap_backend

    assert resolve_bootstrap_backend({}) == "custom"
    assert resolve_bootstrap_backend({"bootstrap_backend": "arch"}) == "arch"
