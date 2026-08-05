"""SPA role assignment helper: HAPPO-as-benchmark vs claim-path claimant."""
from __future__ import annotations

from mascotrl.eval.publication import hansen_spa_with_roles, spa_role_assignment


def test_spa_role_assignment_default_benchmark() -> None:
    roles = spa_role_assignment(happo_as_claimant=False)
    assert roles["benchmark_role"] == "happo"
    assert roles["hansen_benchmark"] == "happo"
    assert roles["hansen_rival"] == "panel"
    assert roles["claimant"] is None
    assert roles["happo_as_claimant"] is False


def test_spa_role_assignment_claim_path_claimant() -> None:
    roles = spa_role_assignment(happo_as_claimant=True)
    assert roles["benchmark_role"] == "panel"
    assert roles["claimant"] == "happo"
    assert roles["hansen_benchmark"] == "panel_member"
    assert roles["hansen_rival"] == "happo"
    assert roles["happo_as_claimant"] is True


def test_hansen_spa_with_roles_stamps_claimant() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    n = 80
    happo = rng.standard_normal(n) * 0.01 + 0.002
    rivals = {
        "a": rng.standard_normal(n) * 0.01,
        "b": rng.standard_normal(n) * 0.01,
        "c": rng.standard_normal(n) * 0.01,
    }
    spa = hansen_spa_with_roles(
        happo.tolist(),
        rivals,
        happo_as_claimant=True,
        n_boot=49,
        seed=0,
    )
    assert spa["claimant"] == "happo"
    assert spa["benchmark_role"] == "panel"
    assert spa.get("ok") is True or "spa_p" in spa
