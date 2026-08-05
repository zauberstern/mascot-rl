"""Baselines peers for research_alpha (Zhang/Moody polarity)."""
from __future__ import annotations

import math

import numpy as np

from mascotrl.eval.research_alpha_baselines import (
    equal_weight_sharpe,
    policy_beats_random,
    random_baseline_sharpe,
    research_baselines_from_returns,
    sign_lag_return_sharpe,
)


def test_baselines_finite_and_named() -> None:
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.01, size=(80, 5))
    peers = research_baselines_from_returns(rets, seed=0)
    for name in ("random", "equal_weight", "long", "sign_lag"):
        assert name in peers
        assert math.isfinite(float(peers[name]["sharpe"]))
    assert math.isfinite(random_baseline_sharpe(rets, seed=1))
    assert math.isfinite(equal_weight_sharpe(rets))
    assert math.isfinite(sign_lag_return_sharpe(rets))


def test_random_baseline_not_absurd_vs_ew() -> None:
    """Static random sleeve should be same order as EW, not Sharpe~50 theater."""
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0003, 0.01, size=(126, 8))
    r = abs(float(random_baseline_sharpe(rets, seed=0)))
    e = abs(float(equal_weight_sharpe(rets)))
    assert r < 20.0
    assert e < 20.0


def test_kill_polarity() -> None:
    assert policy_beats_random(0.1, 0.2) is False
    assert policy_beats_random(0.3, 0.2) is True


def test_friction_matched_ew_differs_from_gross() -> None:
    """W1.2: with factors+friction, EW Sharpe must use parity harness (after-cost)."""
    from mascotrl.eval.friction import FrictionSpec

    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.01, size=(120, 6))
    fac = rng.normal(0.0, 0.01, size=(120, 4))
    fric = FrictionSpec(equity_bps=25.0, impact_c_eq=1.0)
    gross = float(equal_weight_sharpe(rets))
    net = float(equal_weight_sharpe(rets, factors=fac, friction=fric))
    assert math.isfinite(gross) and math.isfinite(net)
    # Positive costs must pull net Sharpe below (or equal only if zero turnover,
    # which EW rebalanced daily is not under monthly mask — still finite).
    assert net != gross or abs(gross) < 1e-9
    peers = research_baselines_from_returns(rets, seed=0, factors=fac, friction=fric)
    assert math.isfinite(float(peers["long"]["sharpe"]))
    assert abs(float(peers["long"]["sharpe"]) - net) < 1e-9


def test_baselines_accept_explicit_cadence_with_rebalance_mask() -> None:
    from mascotrl.eval.friction import FrictionSpec

    rng = np.random.default_rng(4)
    rets = rng.normal(0.0005, 0.01, size=(40, 4))
    factors = rng.normal(0.0, 0.01, size=(40, 4))
    rebalance_mask = np.zeros(40, dtype=bool)
    rebalance_mask[::5] = True

    peers = research_baselines_from_returns(
        rets,
        seed=0,
        factors=factors,
        friction=FrictionSpec(equity_bps=10.0, impact_c_eq=0.0),
        rebalance_mask=rebalance_mask,
        cadence="weekly",
    )

    assert set(peers) == {"random", "equal_weight", "long", "sign_lag"}
