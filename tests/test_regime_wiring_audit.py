"""Wiring audit for regime detectors vs campaign consumers."""
from __future__ import annotations

from pathlib import Path

from mascotrl.eval.regime_wiring_audit import (
    FIGURE_CRITICAL_CONNECTED,
    audit_regime_wiring,
)


ROOT = Path(__file__).resolve().parents[1]


def test_audit_returns_expected_row_ids() -> None:
    report = audit_regime_wiring(ROOT)
    assert report["status"] == "ok"
    ids = {row["id"] for row in report["rows"]}
    for need in (
        "macro_labels_to_behavior",
        "turbulence_to_behavior",
        "hmm_non_test_callers",
        "regime_labels_parquet_readers",
        "per_regime_sharpe_callers",
        "env_features_regime_ids",
        "policy_mode_vs_labels",
        "burst_cherrypick_regime",
        "fixed_share_turbulence_trigger",
        "macro_schedule_tau_yaml",
        "spectrum_happo_macro_series",
        "fioracle_macro_enabled_yaml",
        "seal_to_regime_desk",
    ):
        assert need in ids


def test_seal_to_regime_desk_connected() -> None:
    report = audit_regime_wiring(ROOT)
    by_id = {row["id"]: row for row in report["rows"]}
    assert by_id["seal_to_regime_desk"]["status"] == "connected"
    assert any(
        "assemble_regime_desk.py" in p for p in by_id["seal_to_regime_desk"]["paths"]
    )


def test_confirmatory_critical_paths_connected() -> None:
    report = audit_regime_wiring(ROOT)
    by_id = {row["id"]: row for row in report["rows"]}
    for rid in FIGURE_CRITICAL_CONNECTED:
        assert by_id[rid]["status"] == "connected", rid
    assert report["confirmatory_critical_pass"] is True


def test_hmm_and_per_regime_sharpe_connected() -> None:
    report = audit_regime_wiring(ROOT)
    by_id = {row["id"]: row for row in report["rows"]}
    assert by_id["hmm_non_test_callers"]["status"] == "connected"
    assert any("regime_scorecard.py" in p for p in by_id["hmm_non_test_callers"]["paths"])
    assert by_id["per_regime_sharpe_callers"]["status"] == "connected"
    assert any(
        "regime_scorecard.py" in p for p in by_id["per_regime_sharpe_callers"]["paths"]
    )
    assert by_id["fixed_share_turbulence_trigger"]["status"] == "connected"
    assert any(
        "assemble_regime_desk.py" in p
        for p in by_id["fixed_share_turbulence_trigger"]["paths"]
    )


def test_burst_regime_is_naming_collision() -> None:
    report = audit_regime_wiring(ROOT)
    by_id = {row["id"]: row for row in report["rows"]}
    assert by_id["burst_cherrypick_regime"]["status"] == "naming_collision"


def test_fioracle_enabled_yaml_none() -> None:
    report = audit_regime_wiring(ROOT)
    by_id = {row["id"]: row for row in report["rows"]}
    assert by_id["fioracle_macro_enabled_yaml"]["status"] in (
        "disconnected",
        "none",
    )
    assert by_id["fioracle_macro_enabled_yaml"]["n_matches"] == 0


def test_intentional_non_connections_match_audit() -> None:
    from mascotrl.eval.regime_wiring_audit import INTENTIONAL_NON_CONNECTIONS

    report = audit_regime_wiring(ROOT)
    by_id = {row["id"]: row for row in report["rows"]}
    listed = {e["id"]: e["expected_status"] for e in report["intentional_non_connections"]}
    for rid, expected in INTENTIONAL_NON_CONNECTIONS:
        assert listed[rid] == expected
        actual = by_id[rid]["status"]
        if expected == "none":
            assert actual in ("none", "disconnected")
        else:
            assert actual == expected, (rid, actual, expected)


def test_policy_mode_and_env_ids_intentional() -> None:
    report = audit_regime_wiring(ROOT)
    by_id = {row["id"]: row for row in report["rows"]}
    assert by_id["env_features_regime_ids"]["status"] == "disconnected"
    assert (
        by_id["policy_mode_vs_labels"]["status"]
        == "connected_to_train_disconnected_from_labels"
    )
