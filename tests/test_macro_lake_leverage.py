"""Macro lake leverage inventory vs productive consumers."""
from __future__ import annotations

from pathlib import Path

from mascotrl.data.fioracle_macro import DEFAULT_SERIES, FIORACLE_FEATURE_COLUMNS
from mascotrl.data.paths import MASCOTRL_ROOT
from mascotrl.eval.macro_lake_leverage import (
    PRODUCTIVE_GAPS,
    build_macro_lake_leverage,
)


def test_fioracle_present_and_yield_2y_loaded_dead() -> None:
    report = build_macro_lake_leverage(repo_root=MASCOTRL_ROOT)
    assert report["status"] in ("ok", "partial", "unavailable")
    by = {a["id"]: a for a in report["assets"]}
    if report["status"] == "unavailable":
        assert by["fioracle_raw"]["present_frac"] == 0.0
        assert by["yield_2y"]["status"] == "unavailable"
        return
    assert by["fioracle_raw"]["present_frac"] == 1.0
    assert by["yield_2y"]["status"] == "loaded_dead"
    assert by["epu_z_252"]["status"] == "optional_eval"
    assert by["gpri_z_252"]["status"] == "optional_eval"
    assert by["unemployment_yoy_chg"]["status"] == "engineered_idle"


def test_regime_label_inputs_optional_eval() -> None:
    report = build_macro_lake_leverage(repo_root=MASCOTRL_ROOT)
    by = {a["id"]: a for a in report["assets"]}
    for rid in ("vix_level", "hy_oas_level", "inflation_yoy_level"):
        assert by[rid]["status"] == "optional_eval"


def test_behavior_regressors_partial() -> None:
    report = build_macro_lake_leverage(repo_root=MASCOTRL_ROOT)
    m = report["metrics"]
    assert m["behavior_regressors_used"] == 5
    assert m["behavior_regressors_available"] >= 5
    assert 0.0 < m["fioracle_feature_consumer_frac"] < 1.0
    assert m["regime_label_inputs_used"] == 3


def test_productive_gaps_ranked() -> None:
    report = build_macro_lake_leverage(repo_root=MASCOTRL_ROOT)
    gaps = report["productive_gaps"]
    assert len(gaps) >= 4
    assert gaps[0]["id"] == PRODUCTIVE_GAPS[0]["id"]
    assert "target_files" in gaps[0]


def test_happo_gap_open_by_design() -> None:
    report = build_macro_lake_leverage(repo_root=MASCOTRL_ROOT)
    by = {g["id"]: g for g in report["productive_gaps"]}
    assert by["happo_macro_series_inject"]["status"] == "open_by_design"
    assert PRODUCTIVE_GAPS[-1]["id"] == "happo_macro_series_inject"
    assert PRODUCTIVE_GAPS[-1]["status"] == "open_by_design"


def test_quarantined_not_in_maximize_denominator() -> None:
    report = build_macro_lake_leverage(repo_root=MASCOTRL_ROOT)
    den = set(report["denominator_ids"])
    for q in ("compustat", "ibes", "lseg_p3", "jkp"):
        assert q not in den


def test_missing_usb_is_unavailable_not_crash(tmp_path: Path) -> None:
    report = build_macro_lake_leverage(
        repo_root=MASCOTRL_ROOT,
        usb_lake_root=tmp_path / "no_such_usb",
    )
    by = {a["id"]: a for a in report["assets"]}
    assert by["usb_cboe_vix_designed"]["status"] == "unavailable"
    assert by["usb_interest_rate_designed"]["status"] == "unavailable"


def test_default_series_and_feature_columns_stable() -> None:
    assert len(DEFAULT_SERIES) == 8
    assert len(FIORACLE_FEATURE_COLUMNS) == 11
    assert "yield_2y" in DEFAULT_SERIES
    assert "yield_2y" not in FIORACLE_FEATURE_COLUMNS
    assert "yield_2y_level" not in FIORACLE_FEATURE_COLUMNS
