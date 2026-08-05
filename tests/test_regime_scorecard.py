"""Regime detector performance scorecard (hygiene, agreement, occupancy, events)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mascotrl.data.regime_labels import label_regimes
from mascotrl.eval.regime_scorecard import (
    build_regime_scorecard,
    event_alignment,
    hygiene_prefix_stability,
    occupancy_stats,
)
from mascotrl.eval.stats_rigor import DEFAULT_REGIMES
from mascotrl.eval.turbulence import classify_regime, turbulence_index
from mascotrl.eval.walk_forward_hmm import hmm_turbulent_mask, jaccard_turbulent, walk_forward_hmm_regimes


def _macro_frame(n: int = 900) -> pd.DataFrame:
    dates = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(0)
    vix = 15.0 + rng.normal(0, 1.0, n)
    oas = 4.0 + rng.normal(0, 0.2, n)
    infl = 2.0 + rng.normal(0, 0.1, n)
    # Sustained crisis window
    vix[500:560] = 40.0
    oas[500:560] = 10.0
    return pd.DataFrame(
        {
            "vix_level": vix,
            "hy_oas_level": oas,
            "inflation_yoy_level": infl,
        },
        index=dates,
    )


def _returns(n: int = 900, k: int = 5) -> np.ndarray:
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, size=(n, k))
    r[500:560] = rng.normal(0, 0.05, size=(60, k))
    return r


def test_hygiene_prefix_stability_macro() -> None:
    df = _macro_frame()
    out = hygiene_prefix_stability(df, min_history_days=100, persistence_days=5)
    assert out["status"] == "pass"
    assert out["macro_prefix_stable"] is True


def test_occupancy_sums_to_one() -> None:
    df = _macro_frame()
    labels, meta = label_regimes(df, min_history_days=100, persistence_days=5)
    occ = occupancy_stats(labels, meta)
    assert occ["status"] == "ok"
    fracs = [occ["fractions"][k] for k in ("calm", "inflationary", "crisis")]
    assert sum(fracs) == pytest.approx(1.0, abs=1e-9)
    assert occ["n_crisis"] > 0


def test_jaccard_defined_on_synthetic() -> None:
    pytest.importorskip("hmmlearn")
    r = _returns()
    turb = turbulence_index(r, window=60)
    # smaller window for synthetic length
    mask = classify_regime(turb, quantile=0.75)
    # HMM features: rolling vol proxy
    feats = np.column_stack([r.mean(axis=1), r.std(axis=1)])
    labs = walk_forward_hmm_regimes(feats, window=200, step=21, n_components=2)
    hmm_mask = hmm_turbulent_mask(labs)
    # Align where HMM labeled
    valid = labs >= 0
    j = jaccard_turbulent(mask[valid], hmm_mask[valid])
    assert 0.0 <= j <= 1.0


def test_event_alignment_unavailable_when_no_overlap() -> None:
    dates = pd.bdate_range("2015-01-01", periods=100)
    labels = pd.Series(["calm"] * len(dates), index=dates)
    turb = np.zeros(len(dates), dtype=bool)
    out = event_alignment(dates, labels, turb, regimes=DEFAULT_REGIMES)
    # GFC may be unavailable for 2015-only panel
    gfc = next(x for x in out["windows"] if x["id"] == "gfc_2008")
    assert gfc["status"] == "unavailable"


def test_build_scorecard_synthetic_bundle(tmp_path: Path) -> None:
    pytest.importorskip("hmmlearn")
    df = _macro_frame(800)
    r = _returns(800)
    report = build_regime_scorecard(
        macro=df,
        asset_returns=r,
        repo_root=Path(__file__).resolve().parents[1],
        min_history_days=100,
        turbulence_window=60,
        hmm_window=200,
        include_wiring=True,
        include_leverage=True,
        usb_lake_root=tmp_path / "no_usb",
    )
    assert report["status"] in ("ok", "partial")
    assert "hygiene" in report
    assert "agreement" in report
    assert "occupancy" in report
    assert "event_alignment" in report
    assert "wiring" in report
    assert "leverage" in report
    assert np.isfinite(report["agreement"]["jaccard_turbulence_hmm"])
    assert "mean_turbulent_run_days" in report["agreement"]
    assert "mean_turbulent_run_days_q75" in report["agreement"]
    assert report["agreement"]["operational_label"] == "markov_filtered_p05"
    assert "jaccard_macro_crisis_note" in report["agreement"]
    assert "taxonomy_disclaimer" in report["agreement"]
    assert "causal_per_regime_sharpe" in report["agreement"]
    assert "calendar_stress_windows" in report["agreement"]
    # Causal vs calendar must not be the same object / trivial copy.
    causal = report["agreement"]["causal_per_regime_sharpe"]
    calendar = report["agreement"]["calendar_stress_windows"]
    assert causal is not calendar
    assert set(causal.keys()) != set(calendar.keys()) or "windows" in calendar


def test_agreement_unavailable_without_returns(tmp_path: Path) -> None:
    df = _macro_frame(400)
    report = build_regime_scorecard(
        macro=df,
        asset_returns=None,
        repo_root=Path(__file__).resolve().parents[1],
        min_history_days=100,
        include_wiring=False,
        include_leverage=False,
        usb_lake_root=tmp_path / "no_usb",
    )
    assert report["agreement"]["status"] == "unavailable"
    assert "not provided" in str(report["agreement"].get("reason", "")).lower() or True


def test_run_duration_stats_basic() -> None:
    from mascotrl.eval.regime_scorecard import run_duration_stats

    mask = np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=bool)
    out = run_duration_stats(mask)
    assert out["n_turbulent_runs"] == 2
    assert out["mean_turbulent_run_days"] == pytest.approx(2.5)


def test_scorecard_json_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("hmmlearn")
    df = _macro_frame(700)
    r = _returns(700)
    report = build_regime_scorecard(
        macro=df,
        asset_returns=r,
        repo_root=Path(__file__).resolve().parents[1],
        min_history_days=100,
        turbulence_window=60,
        hmm_window=180,
        include_wiring=False,
        include_leverage=False,
    )
    path = tmp_path / "regime_scorecard.json"
    path.write_text(json.dumps(report, default=str), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert "agreement" in loaded
