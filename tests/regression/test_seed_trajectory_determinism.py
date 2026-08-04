"""Same seed must produce identical final policy weights."""
from __future__ import annotations

import pytest
import torch

from tests.regression._toy_train import run_toy_train


@pytest.mark.slow
@pytest.mark.regression
def test_same_seed_same_trajectory(torch_deterministic):
    """Two full train runs with identical seed produce identical final weights."""
    w1 = run_toy_train(seed=42)
    w2 = run_toy_train(seed=42)
    torch.testing.assert_close(w1, w2, atol=0, rtol=0)
