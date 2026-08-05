"""CRUCIBLE discriminability gates G1/G2/G3."""
from __future__ import annotations

import numpy as np
import pytest

from mascotrl.data.crucible import (
    CrucibleSpec,
    feasible_action_diversity_probe,
    select_universe_crucible,
    structure_participation_gate,
    transfer_coefficient_probe,
)
from mascotrl.eval.friction import FrictionSpec


def _uniform_projector(a):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    k = a.size
    return np.full(k, 1.0 / k, dtype=np.float64)


def _softmax_projector(a):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    z = a - np.max(a)
    e = np.exp(z)
    return e / e.sum()


def _null_signal_projector(a):
    """Map every action to equal weight (kills transfer)."""
    return _uniform_projector(a)


def test_g1_fails_on_uniform_projector():
    secids = list(range(20))
    out = feasible_action_diversity_probe(
        secids,
        _uniform_projector,
        n_draws=64,
        rng=np.random.default_rng(0),
        spec=CrucibleSpec(k=20, g1_l1_floor=0.08, g1_entropy_gap_floor=0.60),
    )
    assert out["g1_pass"] is False
    assert out["g1_feasible_l1_vs_ew"] < 0.08
    assert out["g1_entropy_gap"] < 0.60


def test_g1_passes_on_softmax_like_projector():
    secids = list(range(20))
    # Amplify logits so softmax is peaked (large entropy gap + L1 vs EW)
    def peaked(a):
        return _softmax_projector(np.asarray(a, dtype=np.float64) * 8.0)

    out = feasible_action_diversity_probe(
        secids,
        peaked,
        n_draws=128,
        rng=np.random.default_rng(1),
        spec=CrucibleSpec(k=20, g1_l1_floor=0.08, g1_entropy_gap_floor=0.60),
    )
    assert out["g1_pass"] is True
    assert out["g1_feasible_l1_vs_ew"] >= 0.08
    assert out["g1_entropy_gap"] >= 0.60


def test_g1_uses_entropy_gap_not_raw_entropy():
    secids = list(range(16))
    out = feasible_action_diversity_probe(
        secids,
        _uniform_projector,
        n_draws=32,
        rng=np.random.default_rng(2),
        spec=CrucibleSpec(k=16, g1_l1_floor=0.0, g1_entropy_gap_floor=0.60),
    )
    # Maximal entropy => gap near 0 => fail even with L1 floor disabled
    assert out["g1_action_entropy"] == pytest.approx(float(np.log(16)), abs=1e-6)
    assert out["g1_entropy_gap"] < 0.05
    assert out["g1_pass"] is False


def test_g2_fails_when_projector_nulls_signal():
    secids = list(range(12))
    rng = np.random.default_rng(3)
    signal = rng.normal(size=len(secids))
    out = transfer_coefficient_probe(
        secids,
        signal,
        _null_signal_projector,
        spec=CrucibleSpec(k=12, g2_tc_floor=0.35),
    )
    assert out["g2_pass"] is False
    assert abs(out["g2_tc_post_projection"]) < 0.35


def test_g3_fails_when_5pct_below_floor():
    secids = list(range(10))
    t = 80
    rng = np.random.default_rng(4)
    # Pure noise: ridge cannot beat EW after costs
    returns = pd_returns = __import__("pandas").DataFrame(
        rng.normal(0, 0.01, size=(t, len(secids))), columns=secids
    )
    adv = __import__("pandas").DataFrame(
        np.full((t, len(secids)), 1e9), columns=secids
    )
    out = structure_participation_gate(
        secids,
        returns,
        adv,
        FrictionSpec(impact_c_eq=5.0, execution_impact_coef=5.0),
        ladder=(0.01, 0.05, 0.10),
        spec=CrucibleSpec(k=10, g3_sharpe_floor=0.10),
    )
    # Force failure path: if noise happens to pass, inject known-bad diagnostics
    if out["g3_ridge_minus_ew_sharpe_5pct"] >= 0.10:
        out = dict(out)
        out["g3_ridge_minus_ew_sharpe_5pct"] = 0.01
        out["g3_pass"] = (
            out["g3_ridge_minus_ew_sharpe_5pct"] >= 0.10
            and out["g3_ridge_minus_ew_sharpe_1pct"] >= 0.0
            and out["g3_ridge_minus_ew_sharpe_10pct"] >= 0.0
        )
    assert out["g3_pass"] is False


def test_g3_fails_when_10pct_negative_even_if_5pct_ok():
    from mascotrl.data.crucible import _g3_pass_from_ladder

    assert (
        _g3_pass_from_ladder(
            {"1pct": 0.05, "5pct": 0.15, "10pct": -0.02},
            floor=0.10,
        )
        is False
    )
    assert (
        _g3_pass_from_ladder(
            {"1pct": 0.05, "5pct": 0.15, "10pct": 0.02},
            floor=0.10,
        )
        is True
    )


def test_probes_assert_friction_parity():
    secids = list(range(8))
    train = FrictionSpec(equity_bps=5.0)
    oos = FrictionSpec(equity_bps=10.0)
    with pytest.raises(AssertionError, match="friction parity"):
        feasible_action_diversity_probe(
            secids,
            _softmax_projector,
            n_draws=8,
            rng=np.random.default_rng(0),
            friction_train=train,
            friction_oos=oos,
        )
    with pytest.raises(AssertionError, match="friction parity"):
        transfer_coefficient_probe(
            secids,
            np.ones(8),
            _softmax_projector,
            friction_train=train,
            friction_oos=oos,
        )
    with pytest.raises(AssertionError, match="friction parity"):
        structure_participation_gate(
            secids,
            __import__("pandas").DataFrame(np.zeros((20, 8)), columns=secids),
            __import__("pandas").DataFrame(np.ones((20, 8)) * 1e6, columns=secids),
            train,
            ladder=(0.01, 0.05, 0.10),
            friction_oos=oos,
        )


def test_select_universe_crucible_requires_projector():
    with pytest.raises(ValueError, match="live CMDP projector"):
        select_universe_crucible(
            as_of=__import__("pandas").Timestamp("2020-01-02"),
            pool_secids=[1, 2, 3],
            returns=__import__("pandas").DataFrame(),
            ff4_factors=__import__("pandas").DataFrame(),
            adv_panel=__import__("pandas").DataFrame(),
            amihud_panel=__import__("pandas").DataFrame(),
            surface_panel=__import__("pandas").DataFrame(),
            beta_panel=__import__("pandas").DataFrame(),
            projector=None,
            friction_spec=FrictionSpec(),
        )
