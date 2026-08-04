"""deskorg_v1 companion artifact (joint portfolio only)."""
from __future__ import annotations

import json
from pathlib import Path

from src.reporting.deskorg import build_deskorg_artifact, write_deskorg


def test_deskorg_artifact_honesty_locks(tmp_path: Path) -> None:
    art = build_deskorg_artifact(
        cell_id="eq_K100_multi_happo_mlp_mean_std_cao_deskorg",
        claim_tier="narrative",
        behaviour_path="x_policy_behavior.json",
        decision_trace_path="x_decision_trace.jsonl",
        coordination_proxies={"proj_gap_mean": 0.12, "teamtr_skips_sum": 3.0},
    )
    assert art["schema"] == "deskorg_v1"
    assert art["feeds_capital_gates"] is False
    assert art["feeds_codename_clustering"] is False
    assert art["claim_language"] == "joint_portfolio_coordination_only"
    assert "per_agent" not in json.dumps(art).lower()
    assert art["coordination_proxies"]["policy_loss_last_agent_only"] is True
    out = tmp_path / "cell_deskorg.json"
    write_deskorg(out, art)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["joint_portfolio"]["behaviour_path"].endswith("policy_behavior.json")
