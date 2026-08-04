"""Part D.6: spectrum gates are written into cell artifacts."""
from __future__ import annotations

from pathlib import Path

import yaml


def test_run_cell_writes_gate_verdicts(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_spectrum_campaign as rsc

    def fake_research(cfg, arm, **_kwargs):
        return (
            {
                "path_summary": {"sharpe_mean": 0.8},
                "policy": {"sharpe_mean": 0.8},
                "baselines": {"equal_weight": 0.2, "min_variance": 0.3},
                "cost_ladder": {
                    "break_even_spread_multiplier": 0.5,
                    "cost_source": "om_touch",
                },
                "policy_returns": [0.01] * 40,
                "factors": [[0.0, 0.0, 0.0, 0.0]] * 40,
                "panel_source": "optionmetrics",
                "real_reference_arm_present": True,
                "turnovers": [0.05, 0.06, 0.04],
            },
            None,
        )

    monkeypatch.setattr(rsc, "_run_research_arm", fake_research)
    cfg = {
        "spectrum_cell_id": "gate_wire_cell",
        "algo": "ppo",
        "architecture": "mlp",
        "objective": "differential_sharpe",
        "train_world": "historical",
        "n_assets": 4,
    }
    path = tmp_path / "gate_wire_cell.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    out = rsc.run_cell(path, dry_run=False)
    assert "gate1" in out and "gate2" in out and "gate3" in out
    assert out["gate1"].get("pass") is True
    assert "pass" in out["gate2"]
    assert out["gate3"].get("pass") is True
