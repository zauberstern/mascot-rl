"""Academic BASELINE_NAMES only; zero/random are not peers."""
from __future__ import annotations

import pytest

from mascotrl.eval.alpha_gates import (
    NONSENSE_PEERS,
    apply_rich_baseline_alpha_gate,
    pick_best_baseline,
    rich_baseline_alpha_ok,
)
from mascotrl.reporting.capital_gates import (
    assert_protocol_provenance,
    default_estimand_residuals,
)
from tests.conftest import FLOAT_TOL, capital_gate_pass_extras


def _lake_attestation() -> dict:
    return {
        "single_write_immutable_lake": True,
        "lake_checksum": "test_lake_checksum",
        "estimand_residuals": default_estimand_residuals({}),
    }


def _passing_factor_alpha() -> dict:
    return {"alpha": {"alpha_significant_05": True, "alpha_t_hac": 3.1}}


def _passing_cost_ladder() -> dict:
    return {"break_even_spread_multiplier": 1.4}


def test_nonsense_peers_exclude_zero_random():
    assert "zero" in NONSENSE_PEERS
    assert "random" in NONSENSE_PEERS


def test_pick_best_baseline_ignores_zero_random():
    suite = {
        "summary": {
            "zero": {"mean_pnl": 99.0, "sharpe": 99.0},
            "random": {"mean_pnl": 98.0, "sharpe": 98.0},
            "short_vol_carry": {"mean_pnl": 0.01, "sharpe": 0.5},
            "garch_vol_timing": {"mean_pnl": 0.05, "sharpe": 1.2},
        }
    }
    name, sm = pick_best_baseline(suite)
    assert name == "garch_vol_timing"
    assert sm["mean_pnl"] == pytest.approx(0.05, **FLOAT_TOL)


def test_rich_baseline_alpha_fail_closed_without_baselines():
    ok, meta = rich_baseline_alpha_ok(
        happo_mean=0.1,
        happo_sharpe=2.0,
        baselines=None,
    )
    assert ok is False
    assert meta["baselines_available"] is False
    assert meta["alpha_pending_baselines"] is True


def test_rich_baseline_alpha_requires_beat_best_on_mean_and_sharpe():
    suite = {
        "summary": {
            "short_vol_carry": {"mean_pnl": 0.05, "sharpe": 1.5},
        }
    }
    ok, meta = rich_baseline_alpha_ok(
        happo_mean=0.06,
        happo_sharpe=1.0,
        baselines=suite,
    )
    assert ok is False
    assert meta["sharpe_beats_best_baseline"] is False

    ok2, meta2 = rich_baseline_alpha_ok(
        happo_mean=0.06,
        happo_sharpe=2.0,
        baselines=suite,
    )
    assert ok2 is True
    assert meta2["best_baseline"] == "short_vol_carry"
    assert meta2["edge_vs_best_baseline"] == pytest.approx(0.01)


def test_rich_baseline_alpha_does_not_require_zero():
    suite = {"summary": {"short_vol_carry": {"mean_pnl": 0.01, "sharpe": 0.5}}}
    ok, _ = rich_baseline_alpha_ok(
        happo_mean=0.08,
        happo_sharpe=2.0,
        baselines=suite,
    )
    assert ok is True


def test_apply_rich_baseline_alpha_gate_stamps_hist_and_report():
    report = {
        "historical_oos": {
            "summary": {
                "happo": {"mean_pnl": 0.08, "sharpe": 2.0},
                "zero": {"mean_pnl": 0.0, "sharpe": 0.0},
                "random": {"mean_pnl": 0.02, "sharpe": 0.3},
            },
            "alpha_found_historical": False,
            "alpha_pending_baselines": True,
        },
        "baselines": {
            "summary": {
                "short_vol_carry": {"mean_pnl": 0.01, "sharpe": 0.5},
            }
        },
        "alpha_found": False,
    }
    out = apply_rich_baseline_alpha_gate(report)
    assert out["alpha_found"] is True
    assert out["historical_oos"]["alpha_found_historical"] is True
    assert out["historical_oos"]["alpha_pending_baselines"] is False
    assert out["best_baseline"] == "short_vol_carry"
    assert out["edge_vs_best_baseline"] == pytest.approx(0.07)
    assert "zero" not in out["historical_oos"]["summary"]
    assert "random" not in out["historical_oos"]["summary"]
    assert "edge_vs_random" not in out
    assert "edge_vs_zero" not in out
    assert "sharpe_beats_random" not in out["historical_oos"]


def test_capital_hygiene_blocks_oos_not_above_best_baseline():
    report = {
        "eval_protocol": "pit_optionmetrics_atm_is_oos",
        "historical_oos": {
            "alpha_found_historical": True,
            "sharpe_beats_best_baseline": False,
            "summary": {
                "happo": {"sharpe": 1.0, "mean_pnl": 0.01},
            },
        },
        "baselines": {
            "summary": {
                "short_vol_carry": {"sharpe": 2.0, "mean_pnl": 0.05},
            }
        },
        "best_baseline": "short_vol_carry",
        "edge_vs_best_baseline": -0.04,
        "capital_gates_require_stability": False,
        "capital_gates_require_retrain_wfo": False,
        "factor_alpha": _passing_factor_alpha(),
        "cost_ladder": _passing_cost_ladder(),
        **_lake_attestation(),
    }
    out = assert_protocol_provenance(report)
    fails = out["protocol_gate"]["gate_failures"]
    assert any("best_baseline" in f for f in fails)
    assert not any("random" in f or "zero" in f for f in fails)


def test_capital_hygiene_does_not_require_random_or_zero_peer():
    report = {
        "eval_protocol": "pit_optionmetrics_atm_is_oos",
        "historical_oos": {
            "alpha_found_historical": True,
            "sharpe_beats_best_baseline": True,
            "summary": {
                "happo": {"sharpe": 3.0, "mean_pnl": 0.1},
            },
        },
        "baselines": {
            "summary": {
                "short_vol_carry": {"sharpe": 1.0, "mean_pnl": 0.02},
            }
        },
        "best_baseline": "short_vol_carry",
        "edge_vs_best_baseline": 0.08,
        "capital_gates_require_stability": False,
        "capital_gates_require_retrain_wfo": False,
        "factor_alpha": _passing_factor_alpha(),
        "cost_ladder": _passing_cost_ladder(),
        **_lake_attestation(),
        **capital_gate_pass_extras(),
    }
    out = assert_protocol_provenance(report)
    assert out["protocol_gate"].get("require_sharpe_vs_random") is False
    assert out["protocol_gate"].get("require_sharpe_vs_best_baseline") is True
