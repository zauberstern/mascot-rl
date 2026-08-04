"""emit_ch10_desk_macros reads desk JSON only."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.emit_ch10_desk_macros import emit_desk_macros


def test_emit_desk_macros_from_payload() -> None:
    desk = {
        "synthetic": False,
        "dates": list(range(10)),
        "regret_gap": 1.5,
        "regret_bound": 2.0,
        "alpha": 0.01,
        "k_switches": 5,
        "eta": 0.5,
        "n_experts_active": 6,
        "timeline_source": "seal:usb_kpt10_v3",
        "operational_label": "markov_filtered_p05",
        "diagnostics": {
            "fixed_share_sharpe": 0.5,
            "equal_weight_sharpe": 0.4,
            "oracle_sharpe": 1.2,
            "fs_beats_ew_sharpe": True,
            "best_solo": {"name": "cheetah", "sharpe": 0.6},
            "by_regime": {
                "turbulent": {"fixed_share": {"sharpe": 0.7}},
                "calm": {"fixed_share": {"sharpe": 0.3}},
            },
            "table": {"fixed_share": {"sharpe": 0.5}},
            "alpha_sensitivity": [{"alpha": 0.01, "sharpe": 0.5}],
        },
        "honesty": {"seal_name": "usb_kpt10_v3"},
    }
    out = emit_desk_macros(desk)
    assert out["fixed_share_sharpe"] == 0.5
    assert out["oracle_sharpe"] == 1.2
    assert out["timeline_source"] == "seal:usb_kpt10_v3"
    assert out["T"] == 10
    assert "alignment_jaccard" not in out
    desk["diagnostics"]["mixer_sharpes"] = {"hold_leader_annual": 1.1, "eg_experts": 0.9}
    out2 = emit_desk_macros(desk)
    assert out2["hold_leader_annual_sharpe"] == 1.1
    assert out2["eg_experts_sharpe"] == 0.9
