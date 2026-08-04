from __future__ import annotations

from scripts.build_landed_panel import filter_panel_exclude_near_ew


def test_filter_excludes_low_l1_and_recomputes_twins() -> None:
    panel = {
        "utc": "t",
        "panel_scope": "RC6_* only",
        "rows": [
            {
                "stem": "eq_a_sparse",
                "wave": "RC6",
                "head": "sparse_tilt",
                "l1_vs_ew": 1.8,
                "sharpe": 0.1,
                "archetype": "mixed",
                "alignment_pass": True,
                "has_pb": True,
            },
            {
                "stem": "eq_a_softmax",
                "wave": "RC6",
                "head": "softmax",
                "l1_vs_ew": 0.05,
                "sharpe": 0.9,
                "archetype": "mixed",
                "alignment_pass": True,
                "has_pb": True,
            },
            {
                "stem": "eq_b_sparse",
                "wave": "RC6",
                "head": "sparse_tilt",
                "l1_vs_ew": 1.5,
                "sharpe": 0.0,
                "archetype": "trend_follower",
                "alignment_pass": False,
                "has_pb": True,
            },
            {
                "stem": "eq_b_softmax",
                "wave": "RC6",
                "head": "softmax",
                "l1_vs_ew": 0.08,
                "sharpe": 0.8,
                "archetype": "mixed",
                "alignment_pass": True,
                "has_pb": True,
            },
        ],
    }
    out = filter_panel_exclude_near_ew(panel, min_l1=0.25)
    assert out["n_rows"] == 2
    assert {r["stem"] for r in out["rows"]} == {"eq_a_sparse", "eq_b_sparse"}
    assert out["n_twins_all"] == 0
    assert out["filter"]["n_excluded"] == 2
