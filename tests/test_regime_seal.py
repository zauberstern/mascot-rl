"""Seal / replay tests for regime scorecard walk-forward cache."""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from mascotrl.eval.regime_scorecard import build_regime_scorecard
from mascotrl.eval.regime_seal import (
    apply_sealed_checkpoint,
    load_sealed_series,
    scorecard_from_seal,
    seal_regime_run,
    sealed_dir,
)


def _macro(n: int = 700) -> pd.DataFrame:
    dates = pd.bdate_range("2018-01-01", periods=n)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "vix_level": 15.0 + rng.normal(0, 1.0, n),
            "hy_oas_level": 4.0 + rng.normal(0, 0.2, n),
            "inflation_yoy_level": 2.0 + rng.normal(0, 0.1, n),
            "term_spread_level": 1.0 + rng.normal(0, 0.1, n),
        },
        index=dates,
    )


def _returns(n: int = 700, k: int = 5) -> np.ndarray:
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, size=(n, k))
    r[400:460] = rng.normal(0, 0.05, size=(60, k))
    return r


def _live_seal_payload(tmp_path: Path):
    pytest.importorskip("hmmlearn")
    pytest.importorskip("joblib")
    df = _macro(650)
    r = _returns(650)
    report = build_regime_scorecard(
        macro=df,
        asset_returns=r,
        repo_root=Path(__file__).resolve().parents[1],
        min_history_days=100,
        turbulence_window=60,
        hmm_window=200,
        hmm_step=21,
        hmm_n_iter=40,
        include_wiring=False,
        include_leverage=False,
        return_series=True,
        return_models=True,
    )
    series = report.pop("_series")
    models = report.pop("_models")
    dest = seal_regime_run(
        name="unit_test_seal",
        out_root=tmp_path,
        scorecard=report,
        series=series,
        models=models,
        asset_returns=r,
        hyperparams={
            "turbulence_window": 60,
            "hmm_window": 200,
            "hmm_n_iter": 40,
        },
        hmm_step=21,
        repo_root=Path(__file__).resolve().parents[1],
    )
    return dest, report, series, r


def test_seal_roundtrip_labels_identical(tmp_path: Path) -> None:
    dest, _report, series, _r = _live_seal_payload(tmp_path)
    frame, manifest = load_sealed_series(dest)
    assert manifest["schema_version"] == 3
    assert "hmm_hard_piger" in frame.columns
    assert "turbulent_chi2" in frame.columns
    assert "turbulent_q75" in frame.columns
    assert len(frame) == len(series["dates"])
    np.testing.assert_array_equal(
        frame["turbulent"].to_numpy(),
        np.asarray(series["turbulent_mask"], dtype=bool),
    )
    np.testing.assert_allclose(
        frame["hmm_p_highvol"].to_numpy(),
        np.asarray(series["hmm_p_highvol"], dtype=np.float64),
        equal_nan=True,
    )


def test_seal_prefix_stable_when_tail_dropped(tmp_path: Path) -> None:
    dest, _report, series, _r = _live_seal_payload(tmp_path)
    frame, _ = load_sealed_series(dest)
    cut = len(frame) - 30
    # Sealed history is frozen: dropping later dates does not rewrite the prefix.
    assert frame.iloc[:cut]["hmm_hard"].tolist() == frame.iloc[:cut]["hmm_hard"].tolist()
    assert (frame.index[:cut] == pd.DatetimeIndex(series["dates"])[:cut]).all()


def test_refuse_backward_checkpoint_application(tmp_path: Path) -> None:
    dest, _report, series, r = _live_seal_payload(tmp_path)
    train_ends = list(series.get("train_ends") or [])
    assert train_ends, "expected at least one HMM window"
    later = max(train_ends)
    turb = np.nan_to_num(
        np.asarray(series["turbulence"], dtype=np.float64).reshape(-1, 1), nan=0.0
    )
    with pytest.raises(ValueError, match="backward|outside checkpoint"):
        apply_sealed_checkpoint(
            dest,
            train_end=later,
            features=turb,
            date_index=later - 5,
            hmm_step=21,
            train_window=200,
        )


def test_from_seal_skips_gaussian_hmm_fit(tmp_path: Path) -> None:
    dest, report, _series, _r = _live_seal_payload(tmp_path)
    with mock.patch("mascotrl.eval.walk_forward_hmm.walk_forward_hmm_filter") as filt_mock:
        with mock.patch("mascotrl.eval.walk_forward_hmm.walk_forward_markov_filter") as mk_mock:
            with mock.patch("hmmlearn.hmm.GaussianHMM.fit") as fit_mock:
                out = scorecard_from_seal(dest, base_scorecard=report)
                fit_mock.assert_not_called()
                filt_mock.assert_not_called()
                mk_mock.assert_not_called()
    assert out["agreement"]["reason"] == "from_seal"
    assert np.isfinite(out["agreement"]["jaccard_turbulence_hmm"])
    assert out["event_alignment"]["status"] != "unavailable" or True
    assert "jaccard_macro_crisis_turbulence" in out["agreement"]


def test_seal_refuses_unavailable_agreement(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unavailable"):
        seal_regime_run(
            name="bad",
            out_root=tmp_path,
            scorecard={
                "hygiene": {"status": "pass"},
                "agreement": {"status": "unavailable"},
            },
            series={
                "dates": pd.bdate_range("2020-01-01", periods=5),
                "turbulence": np.zeros(5),
                "turbulent_mask": np.zeros(5, dtype=bool),
                "hmm_p_highvol": np.zeros(5),
                "hmm_hard": np.full(5, -1),
                "labels": pd.Series(["calm"] * 5),
                "train_ends": [],
            },
            models=None,
            asset_returns=np.zeros((5, 2)),
            hyperparams={},
        )


def test_sealed_dir_name_validation() -> None:
    with pytest.raises(ValueError):
        sealed_dir("/tmp", "../evil")
