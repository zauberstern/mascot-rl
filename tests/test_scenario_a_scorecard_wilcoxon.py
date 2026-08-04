from __future__ import annotations

import json
from pathlib import Path

from scripts.scenario_a_rescue_scorecard import build_scorecard


def test_scorecard_reads_wilcoxon_and_heads_complete(tmp_path: Path) -> None:
    panel = {
        "utc": "test",
        "n_rows": 10,
        "by_wave": {"RC6": 8, "RC6_HEADS": 9},
        "head_summary": {
            "softmax": {
                "n": 5,
                "med_l1": 0.08,
                "archetypes": {"mixed": 3},
                "n_softmax_exception": 1,
                "n_alignment_pass": 1,
                "n_alignment_fail": 4,
            },
            "sparse_tilt": {
                "n": 5,
                "med_l1": 1.8,
                "archetypes": {"trend_follower": 2, "mixed": 3},
                "n_alignment_pass": 2,
                "n_alignment_fail": 3,
            },
        },
        "n_twins_all": 4,
        "twin_med_delta_l1": 1.5,
        "rows": [],
    }
    wilcoxon = {
        "wilcoxon": {
            "pvalue_one_sided_greater": 1e-5,
            "significant_01": True,
        },
        "f6_status": "supported_on_landed_twins",
    }
    sc = build_scorecard(panel=panel, wilcoxon=wilcoxon, desk=None, figures={}, utc="test")
    by_id = {c["id"]: c for c in sc["claim_table"]}
    assert by_id["F6"]["status"] == "supported"
    assert "1e-05" in by_id["F6"]["evidence"] or "1e-5" in by_id["F6"]["evidence"]
    assert by_id["D4"]["status"] == "partial"
    assert "9/9" in by_id["D4"]["evidence"]
    assert sc["key_panel_numbers"]["wilcoxon_pvalue_greater"] == 1e-5
