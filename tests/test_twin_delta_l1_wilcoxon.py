"""Tests for twin ΔL1 Wilcoxon (Scenario A rescue Fix7)."""
from __future__ import annotations

from scripts.twin_delta_l1_wilcoxon import twin_wilcoxon


def test_wilcoxon_greater_detects_positive_deltas() -> None:
    twins = [
        {"base": f"b{i}", "delta_l1": 1.0 + 0.1 * i, "sparse_l1": 1.5, "softmax_l1": 0.1}
        for i in range(20)
    ]
    report = twin_wilcoxon(twins)
    assert report["n_twins"] == 20
    assert report["wilcoxon"]["pvalue_one_sided_greater"] < 0.01
    assert report["wilcoxon"]["significant_01"] is True
    assert report["f6_status"] == "supported_on_landed_twins"


def test_wilcoxon_too_few() -> None:
    report = twin_wilcoxon([{"delta_l1": 1.0}, {"delta_l1": 2.0}])
    assert report["wilcoxon"]["pvalue_one_sided_greater"] is None
