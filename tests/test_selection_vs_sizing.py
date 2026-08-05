"""Selection versus sizing attribution (CRUCIBLE A.8)."""
from __future__ import annotations

import numpy as np

from mascotrl.eval.policy_diagnostics import selection_vs_sizing_attribution


def test_three_legs_sum_to_total():
    rng = np.random.default_rng(0)
    t = 200
    parent = rng.normal(0.0002, 0.01, size=t)
    cruc = parent + rng.normal(0.0001, 0.002, size=t)
    pol = cruc + rng.normal(0.00015, 0.003, size=t)
    out = selection_vs_sizing_attribution(pol, cruc, parent)
    total_m = out["total_mean"]
    parts_m = out["name_set_mean"] + out["sizing_mean"] + out["interaction_mean"]
    assert abs(parts_m - total_m) < 1e-9
    total_s = out["total_sharpe"]
    parts_s = out["name_set_sharpe"] + out["sizing_sharpe"] + out["interaction_sharpe"]
    assert abs(parts_s - total_s) < 1e-9


def test_pure_selection_zero_sizing():
    rng = np.random.default_rng(1)
    t = 250
    parent = rng.normal(0.0, 0.01, size=t)
    cruc = parent + 0.001
    pol = cruc.copy()  # policy == EW on crucible names
    out = selection_vs_sizing_attribution(pol, cruc, parent)
    assert abs(out["sizing_mean"]) < 1e-12
    assert abs(out["sizing_sharpe"]) < 1e-9


def test_pure_sizing_zero_name_set():
    rng = np.random.default_rng(2)
    t = 250
    parent = rng.normal(0.0, 0.01, size=t)
    cruc = parent.copy()  # same opportunity set as parent
    pol = cruc + 0.002
    out = selection_vs_sizing_attribution(pol, cruc, parent)
    assert abs(out["name_set_mean"]) < 1e-12
    assert abs(out["name_set_sharpe"]) < 1e-9
