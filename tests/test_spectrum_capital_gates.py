"""Capital hygiene gates for spectrum objective / transfer / DSR honesty."""
from __future__ import annotations

from mascotrl.reporting.capital_gates import assert_protocol_provenance


def _base_report(**kwargs):
    r = {
        "eval_protocol": "toy_not_capital",
        "historical_oos": {"alpha_found_historical": False},
    }
    r.update(kwargs)
    return r


def test_objective_primary_critic_only_fails_when_protocol_would_pass():
    # Use a capital-grade protocol with hist alpha so capital_ok starts True,
    # then trip the new spectrum gate.
    report = {
        "eval_protocol": "pit_optionmetrics_atm_is_oos",
        "historical_oos": {
            "alpha_found_historical": True,
            "friction_applied": True,
            "claim_label_stem": "dh_ret_lagdelta",
            "sharpe": 1.0,
            "sharpe_beats_best_baseline": True,
            "edge_vs_best_baseline": 0.1,
        },
        "objective_primary": True,
        "objective_gradient_path": "critic_only",
        "nested_wfo": {"mode": "retrain_per_fold", "positive_fold_rate": 1.0},
        "multiseed": {"sharpe_p05": 1.0},
        "adversarial_iv": {"fragile": False, "sharpe_degradation": 0.0},
        "factor_alpha": {"alpha_significant_05": True, "t_hac": 3.0},
        "cost_ladder": {"break_even_spread_multiplier": 1.0},
        "deflated_sharpe_oos": {"significant_05": True, "n_trials_breakdown": {"cells": 1}},
        "best_baseline": {"name": "ew", "sharpe": 0.0},
        "algorithm_provenance": {"truncation_bootstrap": True},
        "arctic_as_of": "2024-01-01",
        "estimand_residuals": {
            "american_residual": "disclosed",
            "borrow_state": "omitted",
        },
        "finetune_friction_applied": True,
        "transfer_ok": True,
        "transfer_report": {"real_reference_arm_present": True},
        "collapse_guard": {"ok": True},
    }
    out = assert_protocol_provenance(report, require_stability_gates=False, require_factor_alpha=False)
    fails = out.get("protocol_gate", {}).get("gate_failures") or []
    assert "objective_primary_claimed_but_critic_only" in fails


def test_spectrum_cell_requires_transfer_and_dsr_breakdown():
    report = {
        "eval_protocol": "combinatorial_purged_cv",
        "historical_oos": {"alpha_found_historical": False},
        "spectrum_cell_id": "reference",
        "spectrum_promotable": True,
        "deflated_sharpe_oos": {"significant_05": False},
        "algorithm_provenance": {"truncation_bootstrap": True},
        "arctic_as_of": "2024-01-01",
        "estimand_residuals": {
            "american_residual": "disclosed",
            "borrow_state": "omitted",
        },
    }
    out = assert_protocol_provenance(report, require_stability_gates=False, require_factor_alpha=False)
    fails = out.get("protocol_gate", {}).get("gate_failures") or []
    assert "spectrum_promotion_without_transfer_report" in fails
    assert "spectrum_promotion_without_collapse_guard" in fails
    assert "dsr_trial_count_excludes_spectrum_cells" in fails
