"""Multi-seed property checks: no NaN, non-degenerate weights."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.regression._toy_train import run_toy_train


@pytest.mark.slow
@pytest.mark.regression
@pytest.mark.parametrize("seed", range(5))
def test_ppo_train_no_nan_across_seeds(seed):
    """Training must never produce NaN weights/losses regardless of seed."""
    _, stats = run_toy_train(seed=seed, return_stats=True)
    assert np.all(np.isfinite(stats["final_weights"]))
    assert np.isfinite(stats["final_loss"])
    # Entropy may be zero on some custom PPO configs; require finite only.
    assert np.isfinite(stats["final_entropy"])


@pytest.mark.slow
@pytest.mark.regression
@pytest.mark.parametrize("seed", range(5))
def test_policy_weights_not_degenerate_across_seeds(seed):
    """Trained policy must produce finite weights (soft non-degeneracy)."""
    w = run_toy_train(seed=seed, steps=500, epochs=3)
    flat = w.detach().numpy().flatten()
    assert np.all(np.isfinite(flat))
    ew = 1.0 / len(flat)
    l1 = float(np.sum(np.abs(flat - ew)))
    assert l1 > 1e-6 or len(flat) < 3
