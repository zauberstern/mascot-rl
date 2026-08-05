"""Phase A: estimand parity between policy and benchmarks."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from mascotrl.arms import ArmSpec
from mascotrl.eval.friction import FrictionSpec
from mascotrl.eval.parity_harness import (
    ESTIMAND_FIELDS,
    assert_estimand_hash,
    estimand_hash,
    score_equal_weight,
    score_strategy,
)
from mascotrl.eval.residualization import ResidualizerState, fit_ff4_residualizer, freeze_residualizer


def _toy_panel(t: int = 80, k: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.01, size=(t, k))
    # FF4-like factors
    fac = rng.normal(0.0, 0.01, size=(t, 4))
    return rets, fac


def _eq_arm(k: int) -> ArmSpec:
    return ArmSpec(id="eq", option_slots=0, equity_slots=k, delta_mode="off")


def test_estimand_hash_stable_and_required_fields():
    h = estimand_hash(
        friction=FrictionSpec(equity_bps=5.0, impact_c_eq=0.5),
        residualize=True,
        cadence="monthly",
    )
    assert isinstance(h, str) and len(h) == 64
    expected_fields = {
        "friction_spec_id", "equity_bps", "impact_c_eq", "borrow_floor_bps_annual",
        "cost_multiplier", "residualize", "cadence", "version", "scorecard",
        "universe_fingerprint", "residualizer_fold_id", "residualizer_model",
        "mask_fingerprint", "reward", "feature_channels_fingerprint",
    }
    for field in ESTIMAND_FIELDS:
        assert field in expected_fields
    assert set(ESTIMAND_FIELDS) == expected_fields
    h2 = estimand_hash(
        friction=FrictionSpec(equity_bps=5.0, impact_c_eq=0.5),
        residualize=True,
        cadence="monthly",
    )
    assert h == h2
    h3 = estimand_hash(
        friction=FrictionSpec(equity_bps=10.0, impact_c_eq=0.5),
        residualize=True,
        cadence="monthly",
    )
    assert h != h3


def test_estimand_hash_total_net_ignores_residualizer_identity():
    """total_net never subtracts factor exposure; differing residualizer fits
    must not make an otherwise-identical total_net estimand non-uniform."""
    fric = FrictionSpec(equity_bps=5.0)
    r1 = ResidualizerState(fold_id="a", model="ff4", betas=np.zeros(4), factor_names=("m", "s", "h", "u"))
    r2 = ResidualizerState(fold_id="b", model="ff4", betas=np.ones(4), factor_names=("m", "s", "h", "u"))
    h1 = estimand_hash(friction=fric, cadence="monthly", scorecard="total_net", residualizer=r1)
    h2 = estimand_hash(friction=fric, cadence="monthly", scorecard="total_net", residualizer=r2)
    assert h1 == h2
    # residual scorecard DOES bind residualizer identity.
    hr1 = estimand_hash(friction=fric, cadence="monthly", scorecard="residual", residualizer=r1)
    hr2 = estimand_hash(friction=fric, cadence="monthly", scorecard="residual", residualizer=r2)
    assert hr1 != hr2


def test_estimand_hash_requires_explicit_cadence():
    with pytest.raises(ValueError, match="cadence"):
        estimand_hash(friction=FrictionSpec(), cadence="")


def test_estimand_hash_varies_with_scorecard_universe_and_reward():
    fric = FrictionSpec(equity_bps=5.0)
    base = estimand_hash(friction=fric, cadence="monthly", scorecard="total_net")
    resid = estimand_hash(friction=fric, cadence="monthly", scorecard="residual")
    assert base != resid
    with_universe = estimand_hash(
        friction=fric, cadence="monthly", universe=["A", "B"]
    )
    assert with_universe != base
    with_reward = estimand_hash(friction=fric, cadence="monthly", reward="differential_sharpe")
    assert with_reward != base
    with_mask = estimand_hash(
        friction=fric, cadence="monthly", rebalance_mask=np.array([True, False, True])
    )
    assert with_mask != base


def test_score_strategy_returns_dual_series_and_costs():
    rets, fac = _toy_panel()
    k = rets.shape[1]
    arm = _eq_arm(k)
    friction = FrictionSpec(equity_bps=5.0, impact_c_eq=0.5, borrow_floor_bps_annual=25.0)
    y = np.nanmean(rets, axis=1)
    resid = freeze_residualizer(fit_ff4_residualizer(y, fac, fold_id="parity"), "parity")

    def ew_fn(returns_hist, *, t, w_prev, **_kw):
        del returns_hist, t, w_prev
        return np.full(k, 1.0 / k, dtype=np.float64)

    out = score_strategy(
        ew_fn,
        rets,
        factors=fac,
        arm=arm,
        friction=friction,
        residualizer=resid,
    )
    assert "total_net" in out and "residual" in out and "turnover" in out
    assert "cost" in out and "gross" in out
    # Series length matches the number of steps the env actually took, not T;
    # no phantom pre-trade zero-return day is injected (A3).
    assert out["total_net"].shape == out["t_index"].shape
    assert out["residual"].shape == out["t_index"].shape
    assert out["t_index"].size < rets.shape[0]
    assert int(out["t_index"][0]) == 1
    # RC5: env cold-starts at EW, so an EW policy has zero entry turnover/cost.
    assert float(np.nansum(np.abs(out["cost"]))) == pytest.approx(0.0, abs=1e-12)
    assert "estimand_hash" in out and "estimand_hash_residual" in out
    assert out["estimand_hash"] != out["estimand_hash_residual"]
    assert_estimand_hash(
        out["estimand_hash"],
        friction=friction,
        residualize=True,
        cadence="daily",
        residualizer=resid,
    )


def test_equal_weight_through_harness_matches_self_consistency():
    """Same weight function scored twice must be byte-identical."""
    rets, fac = _toy_panel(t=60, k=3, seed=7)
    k = rets.shape[1]
    arm = _eq_arm(k)
    friction = FrictionSpec(equity_bps=5.0, impact_c_eq=0.5)
    y = np.nanmean(rets, axis=1)
    resid = freeze_residualizer(fit_ff4_residualizer(y, fac, fold_id="parity2"), "parity2")

    a = score_equal_weight(rets, factors=fac, arm=arm, friction=friction, residualizer=resid)
    b = score_equal_weight(rets, factors=fac, arm=arm, friction=friction, residualizer=resid)
    np.testing.assert_allclose(a["total_net"], b["total_net"], atol=1e-12)
    np.testing.assert_allclose(a["residual"], b["residual"], atol=1e-12)
    # Dual scorecard: residual removes factor exposure; series need not be identical.
    assert a["total_net"].shape == a["residual"].shape


def test_assert_estimand_hash_fail_closed_on_mismatch():
    friction = FrictionSpec(equity_bps=5.0)
    good = estimand_hash(friction=friction, residualize=True, cadence="monthly")
    assert_estimand_hash(good, friction=friction, residualize=True, cadence="monthly")
    with pytest.raises(AssertionError, match="estimand_hash"):
        assert_estimand_hash(
            "0" * 64,
            friction=friction,
            residualize=True,
            cadence="monthly",
        )


def test_no_trade_weight_fn_yields_zero_pnl_not_equal_weight():
    """A4: an intentional zero-weight vector must not become equal-weight."""
    from mascotrl.eval.benchmark_panel import get_weight_fn

    rets, fac = _toy_panel(t=60, k=4, seed=3)
    k = rets.shape[1]
    arm = _eq_arm(k)
    friction = FrictionSpec(equity_bps=5.0)
    resid = freeze_residualizer(
        fit_ff4_residualizer(np.nanmean(rets, axis=1), fac, fold_id="nt"), "nt"
    )
    out = score_strategy(
        get_weight_fn("no_trade"),
        rets,
        factors=fac,
        arm=arm,
        friction=friction,
        residualizer=resid,
    )
    # Flat book: zero gross every day. RC5 EW cold-start charges one
    # liquidation cost on the first step, then cost stays zero.
    np.testing.assert_allclose(out["gross"], 0.0, atol=1e-12)
    assert float(out["cost"][0]) > 0.0
    np.testing.assert_allclose(out["cost"][1:], 0.0, atol=1e-12)
    np.testing.assert_allclose(out["weights"], 0.0, atol=1e-12)


def test_score_strategy_requires_explicit_cadence_with_mask():
    rets, fac = _toy_panel(t=40, k=3, seed=5)
    k = rets.shape[1]
    arm = _eq_arm(k)
    mask = np.ones(rets.shape[0], dtype=bool)

    def ew_fn(returns_hist, *, t, w_prev, **_kw):
        del returns_hist, t, w_prev
        return np.full(k, 1.0 / k, dtype=np.float64)

    with pytest.raises(ValueError, match="cadence"):
        score_strategy(
            ew_fn,
            rets,
            factors=fac,
            arm=arm,
            rebalance_mask=mask,
        )


def test_policy_and_benchmark_hash_recipes_align_on_total_net():
    """A6: mirror the exact kwargs run_eq_alloc_campaign.py and
    run_research_alpha_cpcv.py use so a future edit to either call site that
    silently drifts the recipe is caught here rather than only at campaign
    run time (which needs real lake data)."""
    fric = FrictionSpec(equity_bps=5.0, impact_c_eq=0.5)
    universe = [100892, 100937, 101183]
    mask = np.array([True, False, False, True, False], dtype=bool)
    cadence = "monthly"

    # Benchmark-panel recipe (score_strategy via score_benchmark_panel).
    bench_h = estimand_hash(
        friction=fric,
        cadence=cadence,
        scorecard="total_net",
        universe=universe,
        rebalance_mask=mask,
        residualizer=freeze_residualizer(
            fit_ff4_residualizer(np.zeros(5), np.zeros((5, 4)), fold_id="eq_alloc_bench"),
            "eq_alloc_bench",
        ),
    )
    # Policy recipe (run_research_alpha_cpcv's policy_hash_total_net).
    policy_h = estimand_hash(
        friction=fric,
        cadence=cadence,
        scorecard="total_net",
        universe=universe,
        rebalance_mask=mask,
    )
    assert bench_h == policy_h, "total_net hash must not depend on residualizer identity"


def test_campaign_stats_require_matching_estimand_hash():
    """Fail-closed helper used before writing stats_table.json."""
    from mascotrl.eval.parity_harness import require_uniform_estimand_hashes

    friction = FrictionSpec(equity_bps=5.0)
    h = estimand_hash(friction=friction, residualize=True, cadence="monthly")
    entries = {
        "equal_weight": {"estimand_hash": h, "sharpe_total_net": 0.5},
        "policy": {"estimand_hash": h, "sharpe_total_net": 0.4},
    }
    require_uniform_estimand_hashes(entries)
    entries["policy"]["estimand_hash"] = "deadbeef" + "0" * 56
    with pytest.raises(AssertionError, match="estimand_hash"):
        require_uniform_estimand_hashes(entries)
