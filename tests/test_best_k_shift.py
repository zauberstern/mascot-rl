"""Best-k-shift hindsight oracle via dynamic programming."""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.best_k_shift import best_k_shift, theoretical_regret_bound


def test_k0_recovers_best_single_expert() -> None:
    losses = np.array(
        [
            [1.0, 0.5, 2.0],
            [1.0, 0.4, 2.0],
            [1.0, 0.6, 2.0],
        ],
        dtype=np.float64,
    )
    path, total = best_k_shift(losses, k=0)
    assert path == [1, 1, 1]
    assert total == pytest.approx(1.5)


def test_k_max_recovers_omniscient_per_step_best() -> None:
    losses = np.array(
        [
            [0.1, 9.0, 9.0],
            [9.0, 0.2, 9.0],
            [9.0, 9.0, 0.3],
            [0.4, 9.0, 9.0],
        ],
        dtype=np.float64,
    )
    t = losses.shape[0]
    path, total = best_k_shift(losses, k=t - 1)
    assert path == [0, 1, 2, 0]
    assert total == pytest.approx(1.0)


def test_limited_k_cannot_switch_every_step() -> None:
    losses = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    # With k=1 switch, cannot hit all four omniscient picks (needs 3 switches).
    path, total = best_k_shift(losses, k=1)
    omni_total = float(losses.min(axis=1).sum())
    assert total > omni_total - 1e-12
    # Number of switches in path <= 1.
    switches = sum(1 for a, b in zip(path, path[1:]) if a != b)
    assert switches <= 1


def test_theoretical_regret_bound_positive() -> None:
    bound = theoretical_regret_bound(n_experts=6, k_switches=10, sequence_length=2500)
    assert bound > 0.0
