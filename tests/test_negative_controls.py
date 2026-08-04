"""Negative-control fail-closed gates."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.eval.research_alpha_cpcv as research_alpha_cpcv
from src.eval.ceiling_arms import zscore_composite_weights
from src.eval.cpcv import CPCVConfig
from src.eval.negative_controls import (
    date_shift_signals,
    degradation_ratio,
    degradation_should_fail,
    negative_control_should_fail,
    permute_signals_across_names,
    policy_level_negative_control_stamp,
    run_negative_controls,
    shuffled_label_control,
    shuffled_return_rows,
)


def test_shuffled_label_destroys_time_order():
    r = np.arange(20, dtype=float).reshape(10, 2)
    s = shuffled_label_control(r, seed=0)
    assert s.shape == r.shape
    assert not np.allclose(s, r)
    s2 = shuffled_return_rows(r, seed=0)
    np.testing.assert_allclose(s, s2)


def test_permute_signals_changes_columns():
    sig = {"a": np.arange(12, dtype=float).reshape(3, 4)}
    out = permute_signals_across_names(sig, seed=1)
    assert out["a"].shape == (3, 4)
    assert sorted(out["a"][0].tolist()) == sorted(sig["a"][0].tolist())


def test_date_shift_moves_mass():
    sig = {"a": np.eye(5)}
    out = date_shift_signals(sig, shift=2)
    assert np.isnan(out["a"][:2]).all()
    np.testing.assert_allclose(out["a"][2:, 0], sig["a"][:3, 0])


def test_run_negative_controls_fail_closed_legacy_abs_floor():
    ok = run_negative_controls(
        policy_sharpe_on_shuffled=0.1,
        policy_sharpe_on_permuted_signals=-0.2,
        policy_sharpe_on_date_shifted=0.0,
    )
    assert ok["pipeline_broken"] is False
    assert ok["verdict_mode"] == "abs_floor"
    bad = run_negative_controls(
        policy_sharpe_on_shuffled=2.0,
        policy_sharpe_on_permuted_signals=0.0,
        policy_sharpe_on_date_shifted=0.0,
    )
    assert bad["pipeline_broken"] is True
    assert negative_control_should_fail(2.0) is True


def test_degradation_ratio_verdict_passes_when_corruption_destroys_edge():
    """Clean residual Sharpe high; corrupted near zero → pipeline OK."""
    out = run_negative_controls(
        control_sharpe_on_shuffled=0.05,
        control_sharpe_on_permuted_signals=-0.02,
        control_sharpe_on_date_shifted=0.01,
        clean_sharpe=1.2,
        max_degradation_ratio=0.5,
    )
    assert out["verdict_mode"] == "degradation_ratio"
    assert out["pipeline_broken"] is False
    assert out["checks"]["permuted_signals"]["degradation_ratio"] < 0.5


def test_degradation_ratio_verdict_fails_when_corruption_preserves_edge():
    out = run_negative_controls(
        control_sharpe_on_shuffled=1.1,
        control_sharpe_on_permuted_signals=0.9,
        control_sharpe_on_date_shifted=1.0,
        clean_sharpe=1.2,
        max_degradation_ratio=0.5,
    )
    assert out["pipeline_broken"] is True
    assert degradation_should_fail(1.0, clean_sharpe=1.2) is True
    assert degradation_ratio(0.6, clean_sharpe=1.2) == pytest.approx(0.5)


def test_policy_level_stamp_reuses_degradation_verdict():
    stamp = policy_level_negative_control_stamp(
        control_sharpe=0.3,
        clean_sharpe=1.2,
        seed=7,
        fold_id=2,
    )
    assert stamp == {
        "sharpe": 0.3,
        "seed": 7,
        "fold_id": 2,
        "degradation_ratio": pytest.approx(0.25),
        "failed": False,
    }


def test_policy_level_control_runs_one_fold_with_permuted_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.bdate_range("2020-01-01", periods=30)
    returns = np.zeros((30, 4))
    factors = np.zeros((30, 4))
    signals = {"alpha": np.tile(np.asarray([1.0, 2.0, 3.0, 4.0]), (30, 1))}
    seen: dict[str, object] = {}

    def fake_train(cfg, rets, fac, train_idx, *, seed):
        seen["fold_id"] = cfg["_fold_id"]
        seen["seed"] = seed
        seen["signals"] = cfg["feature_extras"]["iv_surface"]["alpha"]
        return {"agent": object()}

    monkeypatch.setattr(research_alpha_cpcv, "_train_agent_for_fold", fake_train)
    monkeypatch.setattr(
        research_alpha_cpcv,
        "_roll_test_pnl",
        lambda **kwargs: {
            str(i): {"total_net": value}
            for i, value in enumerate((0.01, -0.005, 0.02, 0.0))
        },
    )

    stamp = research_alpha_cpcv.run_policy_level_negative_control(
        dates,
        returns,
        factors,
        {
            "headline_fill": "pct75",
            "primary_train": "historical_arm_env",
            "n_assets": 4,
        },
        cpcv=CPCVConfig(
            n_splits=3, n_test_groups=1, purge_days=0, embargo_days=0
        ),
        seed=4,
        clean_sharpe=1.0,
        signals=signals,
    )

    assert seen["fold_id"] == stamp["fold_id"]
    assert seen["seed"] == 4 + stamp["fold_id"]
    assert not np.array_equal(seen["signals"], signals["alpha"])
    assert stamp["seed"] == 4
    assert np.isfinite(stamp["sharpe"])


def _ann_sharpe(pnl: np.ndarray) -> float:
    x = np.asarray(pnl, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    sd = float(np.std(x, ddof=1))
    if sd <= 0:
        return float("nan")
    return float(np.sqrt(252.0) * float(np.mean(x)) / sd)


def test_beta_free_pathway_perm_destroys_known_signal():
    """Behavioural: long-short demeaned zscore on a predictive signal works;
    permuting signal columns across names destroys the residual-like edge.
    """
    rng = np.random.default_rng(7)
    t, k = 400, 20
    # Cross-sectional predictive signal: higher signal → higher next return.
    signal = rng.normal(size=(t, k))
    # Idiosyncratic returns driven by lagged signal (cross-section demeaned).
    noise = rng.normal(scale=0.01, size=(t, k))
    returns = np.zeros((t, k))
    returns[1:] = 0.02 * (signal[:-1] - signal[:-1].mean(axis=1, keepdims=True)) + noise[1:]
    signals = {"alpha": signal}

    def _ls_pnl(sig: dict) -> float:
        pnls = []
        for i in range(1, t):
            w = zscore_composite_weights(sig, t=i - 1, long_only=False)
            pnls.append(float(np.dot(w, returns[i])))
        return _ann_sharpe(np.asarray(pnls))

    clean = _ls_pnl(signals)
    perm = _ls_pnl(permute_signals_across_names(signals, seed=0))
    assert clean > 0.5, f"clean long-short edge too weak: {clean}"
    assert abs(perm) < 0.5 * abs(clean), (
        f"permutation failed to destroy edge: clean={clean} perm={perm}"
    )
    verdict = run_negative_controls(
        control_sharpe_on_shuffled=0.0,
        control_sharpe_on_permuted_signals=perm,
        control_sharpe_on_date_shifted=0.0,
        clean_sharpe=clean,
    )
    assert verdict["checks"]["permuted_signals"]["failed"] is False
    assert verdict["pipeline_broken"] is False
