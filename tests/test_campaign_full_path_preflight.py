"""Always-on + slow full-path campaign preflight (no --skip-rl).

Catches late-stage bugs (stats, gates, policy behavior, report macros) that
the --skip-rl smoke never reaches.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _lake_ready() -> bool:
    from src.data.paths import LAKE_ROOT

    return Path(LAKE_ROOT).exists()


def _allowlist_ready() -> bool:
    path = ROOT / "config" / "signal_allowlist.json"
    if not path.is_file():
        return False
    try:
        blob = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return bool(blob.get("allowlist") or [])


def test_full_path_argv_does_not_set_skip_rl() -> None:
    import scripts.run_eq_alloc_campaign as campaign

    ns = campaign.build_arg_parser().parse_args(
        [
            "--k",
            "6",
            "--seeds",
            "0,1",
            "--universe-arm",
            "dyn_liquidity",
            "--force-overwrite",
            "--no-wfo",
            "--out-dir",
            "/tmp/preflight_probe",
        ]
    )
    assert bool(ns.skip_rl) is False
    assert bool(ns.force_overwrite) is True
    assert bool(ns.no_wfo) is True


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("arm", ["dyn_liquidity", "dyn_hrp"])
def test_campaign_full_path_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arm: str
) -> None:
    if not _lake_ready():
        pytest.skip("volsurf_data_lake not mounted")
    if not _allowlist_ready():
        pytest.skip("config/signal_allowlist.json empty or missing")

    import scripts.run_eq_alloc_campaign as campaign

    out_dir = tmp_path / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        "run_eq_alloc_campaign.py",
        "--config",
        "config/workflows/arm_equity.yaml",
        "--k",
        "8",
        "--max-pool",
        "30",
        "--dii-epochs",
        "2",
        "--seeds",
        "0,1",
        "--universe-arm",
        arm,
        "--train-env-steps",
        "800",
        "--train-epochs",
        "1",
        "--min-optimizer-steps-total",
        "4",
        "--min-optimizer-steps",
        "4",
        "--out-dir",
        str(out_dir),
        "--force-overwrite",
        "--no-wfo",
        "--kelly-n-seeds",
        "1",
        "--kelly-refit-every",
        "150",
        "--kelly-epochs",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    campaign.main()

    summary_path = out_dir / "cpcv_path_summary.json"
    assert summary_path.is_file(), f"missing summary for arm={arm}"
    results = json.loads(summary_path.read_text())

    assert results.get("run_config_hash"), "run_config_hash missing"
    assert isinstance(results.get("eval_panel"), dict)
    assert results["eval_panel"].get("periods_per_year") is not None
    confirmatory = results.get("confirmatory") or {}
    assert isinstance(confirmatory.get("path_summary"), dict)
    assert "sharpe_mean" in confirmatory["path_summary"]
    assert "decision_fields" in confirmatory
    assert "gates" in confirmatory
    for bad in (
        "nested_wfo_eq_error",
        "stats_table_error",
        "yaml_honesty_error",
        "policy_behavior_error",
        "negative_controls_errors",
    ):
        assert bad not in results, f"{bad} present: {results.get(bad)}"

    behavior = results.get("policy_behavior") or {}
    arch = (behavior.get("archetype") or {}).get("name")
    if not arch:
        arch = behavior.get("archetype_primary")
    assert arch, "policy_behavior archetype_primary (or archetype.name) required for strict macros"

    numbers_out = tmp_path / f"numbers_{arm}.tex"
