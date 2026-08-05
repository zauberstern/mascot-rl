"""A12: production-entry-point coverage for score_benchmark_panel."""
from __future__ import annotations

import numpy as np

from mascotrl.arms import ArmSpec
from mascotrl.eval.friction import FrictionSpec
from mascotrl.eval.parity_harness import score_benchmark_panel


def _toy_panel(t: int = 80, k: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.01, size=(t, k))
    fac = rng.normal(0.0, 0.01, size=(t, 4))
    return rets, fac


def _eq_arm(k: int) -> ArmSpec:
    return ArmSpec(id="eq", option_slots=0, equity_slots=k, delta_mode="off")


def test_score_benchmark_panel_scores_every_named_benchmark() -> None:
    rets, fac = _toy_panel()
    k = rets.shape[1]
    out = score_benchmark_panel(
        ["equal_weight", "no_trade"],
        rets,
        factors=fac,
        arm=_eq_arm(k),
        friction=FrictionSpec(equity_bps=5.0, impact_c_eq=0.0),
        cadence="daily",
    )
    assert set(out.keys()) == {"equal_weight", "no_trade"}
    for name, scored in out.items():
        assert "total_net" in scored and "residual" in scored
        assert "estimand_hash" in scored


def test_score_benchmark_panel_no_trade_yields_zero_pnl() -> None:
    rets, fac = _toy_panel()
    k = rets.shape[1]
    out = score_benchmark_panel(
        ["no_trade"],
        rets,
        factors=fac,
        arm=_eq_arm(k),
        friction=FrictionSpec(equity_bps=5.0, impact_c_eq=0.0),
        cadence="daily",
    )
    total_net = np.asarray(out["no_trade"]["total_net"], dtype=float)
    # RC5: EW cold-start → first step pays flatten cost; thereafter flat PnL.
    assert np.allclose(total_net[1:], 0.0)
    assert float(total_net[0]) <= 0.0


def test_score_benchmark_panel_distinguishes_estimand_hashes_by_name_pnl() -> None:
    """Different benchmarks realize different pnl paths under the same
    estimand recipe; the harness must not silently collapse them."""
    rets, fac = _toy_panel()
    k = rets.shape[1]
    out = score_benchmark_panel(
        ["equal_weight", "no_trade"],
        rets,
        factors=fac,
        arm=_eq_arm(k),
        friction=FrictionSpec(equity_bps=5.0, impact_c_eq=0.0),
        cadence="daily",
    )
    ew_total = np.asarray(out["equal_weight"]["total_net"], dtype=float)
    nt_total = np.asarray(out["no_trade"]["total_net"], dtype=float)
    assert not np.allclose(ew_total, nt_total)
