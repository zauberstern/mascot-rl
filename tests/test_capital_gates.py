"""Capital hygiene + external risk guard unit tests."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.reporting.capital_gates import (
    assert_protocol_provenance,
    capacity_curve_from_daily,
    default_estimand_residuals,
    write_known_unmodeled_risks,
)
from tests.conftest import capital_gate_pass_extras


def _lake_attestation() -> dict:
    """Single-write lake attestation required for capital-grade protocols (R7)."""
    return {
        "single_write_immutable_lake": True,
        "lake_checksum": "test_lake_checksum",
        "estimand_residuals": default_estimand_residuals({}),
    }


def test_protocol_gate_blocks_synthetic():
    report = {
        "eval_protocol": "synthetic_rbergomi_holdout",
        "alpha_found": True,
        "alpha_found_synthetic_holdout": True,
        "capital_gates_require_stability": False,
        "capital_gates_require_retrain_wfo": False,
    }
    out = assert_protocol_provenance(report)
    assert out["protocol_gate"]["protocol_hygiene_ok"] is False
    assert out["protocol_gate"]["gate_failures"]


def _passing_factor_alpha() -> dict:
    """Significant positive factor-adjusted alpha (W7 gate evidence)."""
    return {"alpha": {"alpha_significant_05": True, "alpha_t_hac": 3.1}}


def _passing_cost_ladder() -> dict:
    """Edge survives beyond a quarter of the quoted half-spread (W5 gate)."""
    return {"break_even_spread_multiplier": 1.4}


def _passing_baselines() -> dict:
    """Rich peer panel so capital hygiene can gate vs best baseline."""
    return {
        "baselines": {
            "summary": {
                "short_vol_carry": {"mean_pnl": 0.01, "sharpe": 0.5},
            }
        },
        "best_baseline": "short_vol_carry",
        "edge_vs_best_baseline": 0.1,
    }


def test_protocol_gate_allows_hist_oos_without_stability_req():
    report = {
        "eval_protocol": "pit_optionmetrics_atm_is_oos",
        "alpha_found": True,
        "historical_oos": {
            "alpha_found_historical": True,
            "sharpe_beats_best_baseline": True,
            "summary": {
                "happo": {"sharpe": 2.0, "mean_pnl": 0.11},
            },
        },
        "capital_gates_require_stability": False,
        "capital_gates_require_retrain_wfo": False,
        "factor_alpha": _passing_factor_alpha(),
        "cost_ladder": _passing_cost_ladder(),
        **_passing_baselines(),
        **_lake_attestation(),
        **capital_gate_pass_extras(),
    }
    out = assert_protocol_provenance(report)
    assert "capital_claim_allowed" not in out
    assert "tradable_claim_allowed" not in out
    assert out["protocol_gate"]["protocol_hygiene_ok"] is True
    assert "protocol_hygiene_ok" in out["protocol_gate"]


def test_protocol_hygiene_fails_when_gate_ladder_incomplete() -> None:
    report = {
        "eval_protocol": "pit_optionmetrics_atm_is_oos",
        "historical_oos": {
            "alpha_found_historical": True,
            "sharpe_beats_best_baseline": True,
            "summary": {"happo": {"sharpe": 2.0, "mean_pnl": 0.11}},
        },
        "capital_gates_require_stability": False,
        "capital_gates_require_retrain_wfo": False,
        "factor_alpha": _passing_factor_alpha(),
        "cost_ladder": _passing_cost_ladder(),
        **_passing_baselines(),
        **_lake_attestation(),
        "collapse_guard": {"ok": True, "collapse_detected": False},
    }
    out = assert_protocol_provenance(report)
    assert "gate_ladder_failed" in out["protocol_gate"]["gate_failures"]


def test_protocol_hygiene_fails_when_collapse_detected() -> None:
    report = {
        "eval_protocol": "pit_optionmetrics_atm_is_oos",
        "historical_oos": {
            "alpha_found_historical": True,
            "sharpe_beats_best_baseline": True,
            "summary": {"happo": {"sharpe": 2.0, "mean_pnl": 0.11}},
        },
        "capital_gates_require_stability": False,
        "capital_gates_require_retrain_wfo": False,
        "factor_alpha": _passing_factor_alpha(),
        "cost_ladder": _passing_cost_ladder(),
        **_passing_baselines(),
        **_lake_attestation(),
        **capital_gate_pass_extras(),
        "collapse_guard": {
            "collapse_detected": True,
            "collapse_failures": ["turnover_below_floor"],
            "ok": False,
        },
    }
    out = assert_protocol_provenance(report)
    assert "collapse_guard_failed" in out["protocol_gate"]["gate_failures"]


def test_protocol_gate_blocks_dsr_not_significant():
    report = {
        "eval_protocol": "pit_optionmetrics_atm_is_oos",
        "historical_oos": {
            "alpha_found_historical": True,
            "sharpe_beats_best_baseline": True,
            "summary": {
                "happo": {"sharpe": 2.0, "mean_pnl": 0.11},
            },
        },
        "capital_gates_require_stability": False,
        "capital_gates_require_retrain_wfo": False,
        "factor_alpha": _passing_factor_alpha(),
        "cost_ladder": _passing_cost_ladder(),
        "deflated_sharpe_oos": {"significant_05": False, "dsr": 0.0},
        **_passing_baselines(),
        **_lake_attestation(),
    }
    out = assert_protocol_provenance(report)
    assert "dsr_not_significant_05" in out["protocol_gate"]["gate_failures"]


def test_protocol_gate_blocks_oos_sharpe_not_above_best_baseline():
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
    assert not any("random" in f for f in fails)


def test_protocol_gate_blocks_missing_factor_alpha():
    """Beating zero/random cannot carry a claim without factor adjustment."""
    report = {
        "eval_protocol": "pit_optionmetrics_atm_is_oos",
        "historical_oos": {"alpha_found_historical": True},
        "capital_gates_require_stability": False,
        "capital_gates_require_retrain_wfo": False,
        "cost_ladder": _passing_cost_ladder(),
    }
    out = assert_protocol_provenance(report)
    assert "factor_alpha_missing" in out["protocol_gate"]["gate_failures"]


def test_protocol_gate_blocks_insignificant_factor_alpha():
    report = {
        "eval_protocol": "pit_optionmetrics_atm_is_oos",
        "historical_oos": {"alpha_found_historical": True},
        "capital_gates_require_stability": False,
        "capital_gates_require_retrain_wfo": False,
        "factor_alpha": {"alpha": {"alpha_significant_05": False, "alpha_t_hac": 0.4}},
        "cost_ladder": _passing_cost_ladder(),
    }
    out = assert_protocol_provenance(report)
    assert any(
        "factor_adjusted_alpha_not_significant" in f
        for f in out["protocol_gate"]["gate_failures"]
    )


def test_protocol_gate_blocks_edge_that_dies_under_costs():
    report = {
        "eval_protocol": "pit_optionmetrics_atm_is_oos",
        "historical_oos": {"alpha_found_historical": True},
        "capital_gates_require_stability": False,
        "capital_gates_require_retrain_wfo": False,
        "factor_alpha": _passing_factor_alpha(),
        # Dies at 10% of the quoted half-spread.
        "cost_ladder": {"break_even_spread_multiplier": 0.10},
    }
    out = assert_protocol_provenance(report)
    assert any(
        "break_even_spread_multiplier" in f
        for f in out["protocol_gate"]["gate_failures"]
    )


def test_protocol_gate_blocks_missing_cost_ladder():
    report = {
        "eval_protocol": "pit_optionmetrics_atm_is_oos",
        "historical_oos": {"alpha_found_historical": True},
        "capital_gates_require_stability": False,
        "capital_gates_require_retrain_wfo": False,
        "factor_alpha": _passing_factor_alpha(),
    }
    out = assert_protocol_provenance(report)
    assert "cost_ladder_missing" in out["protocol_gate"]["gate_failures"]


def test_protocol_gate_blocks_missing_stability():
    report = {
        "eval_protocol": "pit_optionmetrics_atm_is_oos",
        "historical_oos": {"alpha_found_historical": True},
        "capital_gates_require_stability": True,
        "capital_gates_require_retrain_wfo": True,
    }
    out = assert_protocol_provenance(report)
    assert any("wfo" in f or "multiseed" in f or "adversarial" in f for f in out["protocol_gate"]["gate_failures"])


def test_protocol_gate_passes_full_stability():
    report = {
        "eval_protocol": "pit_optionmetrics_nested_wfo_retrain",
        "historical_oos": {
            "alpha_found_historical": True,
            "sharpe_beats_best_baseline": True,
            "summary": {
                "happo": {"sharpe": 2.0, "mean_pnl": 0.11},
            },
        },
        "capital_gates_require_stability": True,
        "capital_gates_require_retrain_wfo": True,
        "nested_wfo": {
            "mode": "retrain_per_fold",
            "positive_fold_rate": 0.8,
            "finetune_friction_applied": True,
            "n_folds": 5,
        },
        "finetune_friction_applied": True,
        "transfer_ok": True,
        "multiseed_oos": {"sharpe_p05": 0.1},
        "adversarial_iv_stress": {
            "sharpe_degradation": 0.2,
            "fragile": False,
        },
        "factor_alpha": _passing_factor_alpha(),
        "cost_ladder": _passing_cost_ladder(),
        **_passing_baselines(),
        **_lake_attestation(),
        **capital_gate_pass_extras(),
    }
    out = assert_protocol_provenance(report)
    assert out["protocol_gate"]["factor_adjusted_alpha_significant"] is True
    assert out["protocol_gate"]["protocol_hygiene_ok"] is True


def test_protocol_gate_blocks_low_wfo_rate():
    report = {
        "eval_protocol": "pit_optionmetrics_nested_wfo_retrain",
        "historical_oos": {"alpha_found_historical": True},
        "capital_gates_require_stability": True,
        "capital_gates_require_retrain_wfo": True,
        "nested_wfo": {"mode": "retrain_per_fold", "positive_fold_rate": 0.4},
        "multiseed_oos": {"sharpe_p05": 0.1},
        "adversarial_iv_stress": {"sharpe_degradation": 0.1, "fragile": False},
    }
    out = assert_protocol_provenance(report)
    assert any("wfo_positive_fold_rate" in f for f in out["protocol_gate"]["gate_failures"])


def test_protocol_gate_blocks_multiseed_p05():
    report = {
        "eval_protocol": "pit_optionmetrics_nested_wfo_retrain",
        "historical_oos": {"alpha_found_historical": True},
        "capital_gates_require_stability": True,
        "capital_gates_require_retrain_wfo": True,
        "nested_wfo": {"mode": "retrain_per_fold", "positive_fold_rate": 0.9},
        "multiseed_oos": {"sharpe_p05": -0.2},
        "adversarial_iv_stress": {"sharpe_degradation": 0.1, "fragile": False},
    }
    out = assert_protocol_provenance(report)
    assert any("multiseed_sharpe_p05" in f for f in out["protocol_gate"]["gate_failures"])


def test_protocol_gate_requires_protocol():
    with pytest.raises(RuntimeError):
        assert_protocol_provenance({"alpha_found": True})


def test_capacity_curve_monotonic_drag():
    pnls = [0.01] * 100
    turns = [0.15] * 100
    curve = capacity_curve_from_daily(pnls, turns, impact_coef=0.05, spread_bps=10.0)
    assert curve["rows"]
    m1 = next(r for r in curve["rows"] if r["aum_multiplier"] == 1.0)
    m16 = next(r for r in curve["rows"] if r["aum_multiplier"] == 16.0)
    assert m16["mean_impact_drag"] > m1["mean_impact_drag"]


def test_known_unmodeled_risks_written(tmp_path: Path):
    p = write_known_unmodeled_risks(tmp_path)
    assert p.is_file()
    text = p.read_text()
    assert "sim_train_rbergomi" in text
    assert "PRODUCTION CEILING" in text or "single_book_k_limit" in text

