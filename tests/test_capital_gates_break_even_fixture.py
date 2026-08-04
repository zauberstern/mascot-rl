"""E-3: hand-computed break-even spread multiplier through gate1 / capital_gates."""
from __future__ import annotations

import math

import pytest

from scripts.run_eq_alloc_campaign import compute_eq_campaign_gates
from src.eval.spectrum_gates import compute_gate1
from src.reporting.capital_gates import assert_protocol_provenance


def _break_even_from_ladder(mid: float, pct75: float) -> float:
    slope = pct75 - mid
    if not (math.isfinite(mid) and math.isfinite(slope) and abs(slope) > 1e-9):
        return float("nan")
    return 0.5 - mid * (0.5 / slope)


def test_break_even_hand_computed_from_fill_ladder() -> None:
    mid, pct75 = 1.024, 1.0
    expected = _break_even_from_ladder(mid, pct75)
    gates = compute_eq_campaign_gates(
        fill_ladder={"mid": mid, "pct75": pct75},
        policy_sharpe=1.0,
        challenger_sharpes={},
        series0=None,
        series0_dates=None,
        panel_dates=["2020-01-02", "2020-01-03"],
        lake_root="/nonexistent",
    )
    be = gates["gate1"]["break_even_spread_multiplier"]
    assert be == pytest.approx(expected)
    assert be == pytest.approx(21.833333333333314)


def test_zero_cost_edge_yields_non_finite_break_even() -> None:
    """Flat ladder (zero Sharpe slope) must not invent a finite multiplier."""
    mid = pct75 = 0.8
    be = _break_even_from_ladder(mid, pct75)
    assert not math.isfinite(be)
    gate1 = compute_gate1({"break_even_spread_multiplier": be})
    assert gate1["pass"] is False


def test_capital_gates_refuses_non_finite_break_even_multiplier() -> None:
    report = {
        "eval_protocol": "pit_optionmetrics_atm_is_oos",
        "historical_oos": {
            "alpha_found_historical": True,
            "summary": {"happo": {"sharpe": 1.0, "mean_pnl": 0.01}},
        },
        "capital_gates_require_stability": False,
        "capital_gates_require_retrain_wfo": False,
        "capital_gates_require_dsr": False,
        "capital_gates_require_sharpe_vs_best_baseline": False,
        "factor_alpha": {"alpha": {"alpha_significant_05": True, "alpha_t_hac": 2.5}},
        "cost_ladder": {"break_even_spread_multiplier": float("nan")},
        "baselines": {"ew": {"sharpe": 0.1, "mean_pnl": 0.0}},
    }
    out = assert_protocol_provenance(report)
    assert any(
        "break_even_spread_multiplier" in f
        for f in out["protocol_gate"]["gate_failures"]
    )
