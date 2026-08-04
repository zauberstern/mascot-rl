"""Campaign-side CRUCIBLE honesty: live CMDP projector, no softmax stub."""
from __future__ import annotations

import numpy as np

from src.policy.cmdp_projector import make_cmdp_projector, soft_simplex_project


def test_make_cmdp_projector_hard_caps_turnover_from_ew():
    cfg = {"projection_mode": "hard", "turnover_limit": 0.15}
    k = 10
    projector = make_cmdp_projector(cfg, k=k)
    # Raw action that softmax would concentrate hard on one name
    raw = np.zeros(k, dtype=np.float64)
    raw[0] = 50.0
    soft = soft_simplex_project(raw)
    hard = np.asarray(projector(raw), dtype=np.float64)
    ew = np.full(k, 1.0 / k, dtype=np.float64)
    soft_turn = float(np.sum(np.abs(soft - ew)))
    hard_turn = float(np.sum(np.abs(hard - ew)))
    assert soft_turn > 0.15 + 1e-9
    assert hard_turn <= 0.15 + 1e-9
    assert not np.allclose(hard, soft)
