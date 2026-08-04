"""RED: decision_fields must expose PREREG arm-selection inputs from seed arts."""
from __future__ import annotations

from scripts.run_eq_alloc_campaign import _aggregate_decision_fields


def test_aggregate_decision_fields_from_seed_arts() -> None:
    seed_arts = [
        {
            "equal_weight_collapse_detected": False,
            "turnover_cap_binding_fraction": 0.1,
            "policy_diagnostics": {"l1_vs_ew_mean": 0.02},
            "path_summary": {"sharpe_mean": 0.9},
        },
        {
            "equal_weight_collapse_detected": True,
            "turnover_cap_binding_fraction": 0.3,
            "policy_diagnostics": {"l1_vs_ew_mean": 0.04},
            "path_summary": {"sharpe_mean": 1.1},
        },
    ]
    sharpes = [0.9, 1.1]
    out = _aggregate_decision_fields(seed_arts, sharpes)
    assert out["equal_weight_collapse_detected_any"] is True
    assert out["equal_weight_collapse_detected_per_seed"] == [False, True]
    assert abs(out["turnover_cap_binding_fraction_mean"] - 0.2) < 1e-9
    assert abs(out["l1_vs_ew_mean"] - 0.03) < 1e-9
    assert out["n_seeds"] == 2
    assert abs(out["sharpe_mean_across_seeds"] - 1.0) < 1e-9
