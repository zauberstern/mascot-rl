"""Part D.4: HAPPO spectrum budget parity + remote cell validation."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import yaml

from scripts.run_spectrum_campaign import resolve_spectrum_budget, run_cell
from scripts.validate_remote_cell import validate_remote_cell


ROOT = Path(__file__).resolve().parents[1]


from tests.conftest import capital_gate_pass_extras


def test_happo_and_ppo_resolve_same_budget_when_not_dispatch_only() -> None:
    base = {
        "train_episodes": 4,
        "horizon": 32,
        "seeds": [0, 1, 2],
        "cpcv_n_splits": 8,
        "cpcv_n_test_groups": 3,
        "claim_tier": "research",
        "architecture": "mlp",
        "objective": "differential_sharpe",
        "train_world": "historical",
    }
    ppo = resolve_spectrum_budget({**base, "algo": "ppo"})
    happo = resolve_spectrum_budget({**base, "algo": "happo", "protocol_tier": "narrative"})
    assert ppo["dispatch_only"] is False
    assert happo["dispatch_only"] is False
    assert ppo["n_episodes"] == happo["n_episodes"] == 4
    assert ppo["horizon"] == happo["horizon"] == 32
    assert ppo["seeds"] == happo["seeds"] == [0, 1, 2]
    assert ppo["cpcv_n_splits"] == happo["cpcv_n_splits"] == 8


def test_dispatch_only_budget_caps_episodes_and_stamps_claim_tier() -> None:
    """Wave 5: dispatch_only cannot look like matched full-scale episodes."""
    screening = resolve_spectrum_budget(
        {
            "algo": "happo",
            "claim_tier": "research",
            "protocol_tier": "screening",
            "train_episodes": 4,
            "horizon": 32,
            "seeds": [0, 1],
        }
    )
    smoke = resolve_spectrum_budget(
        {
            "algo": "happo",
            "claim_tier": "dispatch_only",
            "train_episodes": 4,
            "horizon": 32,
            "seeds": [0, 1],
        }
    )
    assert screening["dispatch_only"] is True
    assert screening["claim_tier"] == "dispatch_only"
    assert screening["n_episodes"] == 2
    assert screening["horizon"] == 6
    assert smoke["dispatch_only"] is True
    assert smoke["claim_tier"] == "dispatch_only"
    assert smoke["n_episodes"] == 2
    assert smoke["horizon"] == 6


def test_timing_probe_writes_json(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    cfg_dir = tmp_path / "cfgs"
    cfg_dir.mkdir()
    cell = {
        "spectrum_cell_id": "probe_cell",
        "algo": "ppo",
        "architecture": "mlp",
        "objective": "differential_sharpe",
        "train_world": "historical",
        "n_assets": 4,
        "train_episodes": 1,
        "horizon": 8,
        "cpcv_n_splits": 3,
        "cpcv_n_test_groups": 1,
    }
    (cfg_dir / "probe_cell.yaml").write_text(yaml.dump(cell), encoding="utf-8")
    out = tmp_path / "out"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_spectrum_campaign.py"),
        "--config-dir",
        str(cfg_dir),
        "--out-dir",
        str(out),
        "--dry-run",
        "--timing-probe",
        "probe_cell",
    ]
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)
    probe = json.loads((out / "timing_probe_probe_cell.json").read_text(encoding="utf-8"))
    for key in (
        "elapsed_s",
        "peak_rss_mb",
        "n_episodes",
        "horizon",
        "n_folds",
        "projected_hours",
        "n_happo_cells",
        "n_seeds",
    ):
        assert key in probe
    assert probe["n_episodes"] == 1
    assert probe["horizon"] == 8
    assert probe["n_folds"] == 3
    assert float(probe["projected_hours"]) >= 0.0


def test_remote_cell_validation_fingerprint_and_provenance() -> None:
    good = {
        "compute_host": "remote",
        "instance_type": "c6i.4xlarge",
        "container_digest": "sha256:abc",
        "requirements_lock_sha256": "deadbeef",
        "crucible_fingerprint": "fp123",
        "spectrum_cell_id": "eq_algo_happo",
    }
    ok = validate_remote_cell(good, expected_fingerprint="fp123")
    assert ok["ok"] is True
    assert ok["artifact"]["compute_host"] == "remote"

    bad_fp = dict(good)
    bad_fp["crucible_fingerprint"] = "other"
    assert validate_remote_cell(bad_fp, expected_fingerprint="fp123")["ok"] is False

    missing_host = dict(good)
    missing_host.pop("compute_host")
    assert validate_remote_cell(missing_host, expected_fingerprint="fp123")["ok"] is False

