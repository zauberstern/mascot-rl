"""Alpha v2 Block E Step 24: G0–G6 gate ladder predicates."""
from __future__ import annotations

import copy

import pytest

from mascotrl.eval.gate_ladder import run_gate_ladder


def _pass_bundle() -> dict:
    """Synthetic bundle that clears every locked G0–G6 predicate."""
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


def test_synthetic_pass_all_gates():
    out = run_gate_ladder(_pass_bundle())
    assert out["pass"] is True
    assert out["any_fail"] is False
    for g in ("G0", "G1", "G2", "G3", "G4", "G5", "G6"):
        assert out["gates"][g]["pass"] is True


@pytest.mark.parametrize(
    "mutate,gate",
    [
        (lambda b: b["g0"].__setitem__("explicit_arm", False), "G0"),
        (lambda b: b["g0"].__setitem__("pit", False), "G0"),
        (lambda b: b["g0"].__setitem__("friction_parity", False), "G0"),
        (lambda b: b["g0"].__setitem__("no_silent_delist", False), "G0"),
        (lambda b: b["g0"].__setitem__("no_rbergomi_ancestry", False), "G0"),
        (lambda b: b["g0"].__setitem__("eq_allowlist", False), "G0"),
        (lambda b: b["g1"].__setitem__("residual_sr", 0.0), "G1"),
        (lambda b: b["g1"].__setitem__("months_positive", 6), "G1"),
        (lambda b: b["g2"].__setitem__("median_residual_sr", 0.24), "G2"),
        (lambda b: b["g2"].__setitem__("p10_residual_sr", 0.0), "G2"),
        (lambda b: b["g3"].__setitem__("p05_sr", 0.0), "G3"),
        (lambda b: b["g3"].__setitem__("median_sr", 0.49), "G3"),
        (lambda b: b["g3"].__setitem__("ensemble_residual_return", 0.0), "G3"),
        (lambda b: b["g3"].__setitem__("bootstrap_ci_low_median", -0.01), "G3"),
        (lambda b: b["g3"].__setitem__("max_seed_profit_share", 0.51), "G3"),
        (lambda b: b["g3"].__setitem__("n_positive_seeds", 14), "G3"),
        (lambda b: b["g4"].__setitem__("path_srs", [0.6, 0.55, 0.7, 0.52, -0.01]), "G4"),
        (lambda b: b["g4"].__setitem__("path_srs", [0.4, 0.45, 0.48, 0.49, 0.47]), "G4"),
        (lambda b: b["g4"].__setitem__("n_positive_combos", 11), "G4"),
        (lambda b: b["g5"].__setitem__("hac_t", 2.99), "G5"),
        (lambda b: b["g5"].__setitem__("dsr", 0.94), "G5"),
        (lambda b: b["g5"].__setitem__("pbo", 0.11), "G5"),
        (lambda b: b["g5"].__setitem__("spa_p", 0.06), "G5"),
        (lambda b: b["g6"].__setitem__("residual_sr_1_5x", 0.24), "G6"),
        (lambda b: b["g6"].__setitem__("alpha_1_5x", 0.0), "G6"),
        (lambda b: b["g6"].__setitem__("n_paths_positive_1_5x", 3), "G6"),
        (lambda b: b["g6"].__setitem__("capacity_alpha_10m", 0.0), "G6"),
        (lambda b: b["g6"].__setitem__("published_1x", False), "G6"),
        (lambda b: b["g6"].__setitem__("published_3x", False), "G6"),
    ],
)
def test_golden_fail_when_any_gate_false(mutate, gate):
    bundle = copy.deepcopy(_pass_bundle())
    mutate(bundle)
    out = run_gate_ladder(bundle)
    assert out["pass"] is False
    assert out["any_fail"] is True
    assert out["gates"][gate]["pass"] is False
