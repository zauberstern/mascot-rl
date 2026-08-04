from __future__ import annotations

from scripts.build_landed_panel import dedupe_rows_by_stem


def test_dedupe_prefers_higher_priority_wave() -> None:
    rows = [
        {"stem": "eq_K100_single_ddpg_mlp_softmax_mtm_pnl", "wave": "LEGACY", "l1_vs_ew": 0.96},
        {"stem": "eq_K100_single_ddpg_mlp_softmax_mtm_pnl", "wave": "RC6", "l1_vs_ew": 1.97},
    ]
    kept, dropped = dedupe_rows_by_stem(rows)
    assert len(kept) == 1
    assert kept[0]["wave"] == "RC6"
    assert kept[0]["l1_vs_ew"] == 1.97
    assert len(dropped) == 1
    assert dropped[0]["wave"] == "LEGACY"


def test_dedupe_keeps_distinct_stems() -> None:
    rows = [
        {"stem": "a", "wave": "RC6"},
        {"stem": "b", "wave": "LEGACY"},
    ]
    kept, dropped = dedupe_rows_by_stem(rows)
    assert len(kept) == 2
    assert dropped == []
