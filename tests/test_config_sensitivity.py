"""Config sensitivity: objectives must differentiate training."""
from __future__ import annotations

import numpy as np
import torch

from src.policy.objective_factory import (
    episode_weights,
    objective_gradient_path_for,
)


def test_episode_weight_modes_produce_distinct_weights():
    g = torch.tensor([-0.2, -0.1, 0.0, 0.05, 0.1, 0.15, 0.2, -0.05])
    modes = ("mean_std_cao", "cvar_ru", "entropic_oce", "smse", "rsqp")
    mats = {}
    for m in modes:
        assert objective_gradient_path_for(m, True) == "episode_weight"
        mats[m] = episode_weights(m, g).detach().numpy()
    # Pairwise: at least one pair differs (actually all should).
    diffs = 0
    keys = list(mats)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if not np.allclose(mats[keys[i]], mats[keys[j]]):
                diffs += 1
    assert diffs >= 4
