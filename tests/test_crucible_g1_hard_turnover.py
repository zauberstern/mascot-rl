"""G1 floors must be achievable under the live hard CMDP turnover projector."""
from __future__ import annotations

import numpy as np

from mascotrl.data.crucible import (
    CrucibleSpec,
    effective_g1_entropy_gap_floor,
    feasible_action_diversity_probe,
)
from mascotrl.policy.cmdp_projector import make_cmdp_projector


def test_g1_hard_turnover_passes_with_configured_soft_floor():
    """YAML gap floor 0.60 is for unconstrained peaked maps; hard tau clamps it."""
    k = 40
    tau = 0.15
    proj = make_cmdp_projector(
        {"projection_mode": "hard", "turnover_limit": tau}, k=k
    )
    spec = CrucibleSpec(k=k, g1_l1_floor=0.08, g1_entropy_gap_floor=0.60)
    out = feasible_action_diversity_probe(
        list(range(k)),
        proj,
        n_draws=256,
        rng=np.random.default_rng(0),
        spec=spec,
        turnover_limit=tau,
    )
    assert out["g1_pass"] is True
    assert out["g1_feasible_l1_vs_ew"] >= 0.08
    assert out["g1_entropy_gap_floor_effective"] < 0.60
    assert out["g1_entropy_gap"] >= out["g1_entropy_gap_floor_effective"]


def test_g1_still_fails_uniform_projector_with_turnover_clamp():
    k = 40
    tau = 0.15

    def uniform(a):
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        return np.full(a.size, 1.0 / a.size, dtype=np.float64)

    spec = CrucibleSpec(k=k, g1_l1_floor=0.08, g1_entropy_gap_floor=0.60)
    out = feasible_action_diversity_probe(
        list(range(k)),
        uniform,
        n_draws=64,
        rng=np.random.default_rng(1),
        spec=spec,
        turnover_limit=tau,
    )
    assert out["g1_pass"] is False
    assert out["g1_feasible_l1_vs_ew"] < 0.08


def test_effective_gap_floor_clamps_under_tau():
    configured = 0.60
    eff = effective_g1_entropy_gap_floor(k=40, turnover_limit=0.15, configured=configured)
    assert 0.0 < eff < configured
    # Unconstrained (no turnover) keeps configured floor.
    assert effective_g1_entropy_gap_floor(
        k=40, turnover_limit=None, configured=configured
    ) == configured
