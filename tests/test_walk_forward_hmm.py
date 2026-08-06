"""Walk-forward Gaussian HMM regime cross-check (leakage-safe)."""
from __future__ import annotations

import numpy as np
import pytest

from mascotrl.eval.walk_forward_hmm import jaccard_turbulent, walk_forward_hmm_regimes


def test_walk_forward_hmm_no_lookahead_on_prefix() -> None:
    pytest.importorskip("hmmlearn")
    rng = np.random.default_rng(0)
    # Two clear vol regimes in a 2-d feature space.
    calm = rng.normal(0.0, 0.01, size=(800, 2))
    storm = rng.normal(0.0, 0.08, size=(200, 2))
    feats = np.vstack([calm, storm, calm[:200]])
    labels = walk_forward_hmm_regimes(
        feats, window=252 * 2, step=21, n_components=2, random_state=42
    )
    assert labels.shape[0] == feats.shape[0]
    # Warmup is NaN / unset (-1).
    assert np.all(labels[: 252 * 2] < 0)
    # Predictions after warmup exist.
    assert np.any(labels[252 * 2 :] >= 0)
    # Prefix stability: truncating future storm must not change early preds.
    cut = 700
    labels_pref = walk_forward_hmm_regimes(
        feats[:cut], window=252 * 2, step=21, n_components=2, random_state=42
    )
    mask = labels_pref >= 0
    np.testing.assert_array_equal(labels[:cut][mask], labels_pref[mask])


def test_jaccard_turbulent_agreement() -> None:
    a = np.array([False, True, True, False])
    b = np.array([False, True, False, False])
    # Intersection {1}, union {1,2} -> 0.5
    assert jaccard_turbulent(a, b) == pytest.approx(0.5)
    assert jaccard_turbulent(a, a) == pytest.approx(1.0)


def _planted_turbulence_series(t: int = 900, seed: int = 3) -> np.ndarray:
    """1-d turbulence-like score with a persistent high-vol mid block."""
    rng = np.random.default_rng(seed)
    x = rng.normal(1.0, 0.3, size=(t, 1))
    a, b = t // 3, t // 3 + 120
    x[a:b] = rng.normal(8.0, 1.5, size=(b - a, 1))
    return x


def test_viterbi_on_block_has_intra_block_lookahead() -> None:
    """Document the leak: model.predict on a future step block peeks inside the block.

    Hand-built HMM: day-0 emission is ambiguous; later days scream high-vol.
    Viterbi on the full block flips day 0 vs Viterbi on day 0 alone.
    """
    pytest.importorskip("hmmlearn")
    from hmmlearn.hmm import GaussianHMM

    model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=1, init_params="")
    model.startprob_ = np.array([0.95, 0.05])
    model.transmat_ = np.array([[0.95, 0.05], [0.05, 0.95]])
    model.means_ = np.array([[0.0], [5.0]])
    model.covars_ = np.array([[1.0], [1.0]])

    # Ambiguous first obs; later days scream high-vol so full-path Viterbi flips day 0.
    block = np.array([[2.5], [5.0], [5.0], [5.0], [5.0], [5.0]], dtype=np.float64)
    pred_full = model.predict(block)
    pred_day0_only = model.predict(block[:1])
    assert pred_full[0] != pred_day0_only[0], (
        "expected Viterbi day-0 label to depend on later block observations"
    )


def test_filter_no_intra_block_lookahead() -> None:
    """Filtered P(s_t|data through t) at day t unchanged if later OOS days are deleted."""
    pytest.importorskip("hmmlearn")
    from mascotrl.eval.walk_forward_hmm import walk_forward_hmm_filter

    x = _planted_turbulence_series()
    window, step = 400, 21
    out_full = walk_forward_hmm_filter(
        x, window=window, step=step, n_components=2, random_state=42, n_iter=80
    )
    # Truncate so the first OOS block has only its first day of future data.
    cut = window + 1
    out_trunc = walk_forward_hmm_filter(
        x[:cut],
        window=window,
        step=step,
        n_components=2,
        random_state=42,
        n_iter=80,
    )
    assert np.isfinite(out_full["p_highvol"][window])
    assert out_full["p_highvol"][window] == pytest.approx(
        out_trunc["p_highvol"][window], rel=0, abs=1e-12
    )
    assert out_full["hard"][window] == out_trunc["hard"][window]


def test_filter_agrees_with_turbulence_better_than_return_mean_hmm() -> None:
    """KPT: HMM on turbulence should Jaccard-agree with threshold more than return-mean HMM."""
    pytest.importorskip("hmmlearn")
    from mascotrl.eval.turbulence import classify_regime, turbulence_index
    from mascotrl.eval.walk_forward_hmm import (
        hmm_turbulent_mask,
        jaccard_turbulent,
        walk_forward_hmm_filter,
        walk_forward_hmm_regimes,
    )

    rng = np.random.default_rng(11)
    t, n = 800, 6
    returns = rng.normal(0.0, 0.01, size=(t, n))
    a, b = 450, 530
    returns[a:b] = rng.normal(0.0, 0.05, size=(b - a, n))
    turb = turbulence_index(returns, window=120, min_names=3)
    turb_mask = classify_regime(turb, quantile=0.75)

    turb_feat = np.asarray(turb, dtype=np.float64).reshape(-1, 1)
    # Leave warmup NaN; filter drops non-finite train rows.
    filt = walk_forward_hmm_filter(
        turb_feat, window=300, step=21, n_components=2, random_state=42, n_iter=60
    )
    # Secondary: return mean/std HMM (old scorecard path).
    feat_ret = np.column_stack(
        [
            np.nanmean(returns, axis=1),
            np.nanstd(returns, axis=1),
        ]
    )
    hmm_ret = walk_forward_hmm_regimes(
        feat_ret, window=300, step=21, n_components=2, random_state=42, n_iter=60
    )
    valid = np.isfinite(turb) & (filt["hard"] >= 0) & (hmm_ret >= 0)
    j_kpt = jaccard_turbulent(turb_mask[valid], filt["hard"][valid].astype(bool))
    j_ret = jaccard_turbulent(
        turb_mask[valid], hmm_turbulent_mask(hmm_ret[valid])
    )
    assert j_kpt > j_ret
    assert j_kpt > 0.15


def test_markov_filter_k_regimes_must_be_two() -> None:
    pytest.importorskip("statsmodels")
    from mascotrl.eval.walk_forward_hmm import walk_forward_markov_filter

    y = np.random.default_rng(0).normal(0, 1, size=400)
    with pytest.raises(ValueError, match="k_regimes must be 2"):
        walk_forward_markov_filter(y, window=200, step=21, k_regimes=3)


def test_markov_filter_no_intra_block_lookahead() -> None:
    pytest.importorskip("statsmodels")
    from mascotrl.eval.walk_forward_hmm import walk_forward_markov_filter

    rng = np.random.default_rng(3)
    t = 500
    x = rng.normal(1.0, 0.3, size=t)
    x[200:320] = rng.normal(8.0, 1.5, size=120)
    window, step = 250, 21
    full = walk_forward_markov_filter(
        x, window=window, step=step, search_reps=2, maxiter=50
    )
    cut = window + 1
    trunc = walk_forward_markov_filter(
        x[:cut], window=window, step=step, search_reps=2, maxiter=50
    )
    assert np.isfinite(full["p_highvol"][window])
    assert full["p_highvol"][window] == pytest.approx(
        trunc["p_highvol"][window], rel=0, abs=1e-8
    )


def test_markov_piger_more_persistent_than_hard() -> None:
    pytest.importorskip("statsmodels")
    from mascotrl.eval.regime_scorecard import run_duration_stats
    from mascotrl.eval.walk_forward_hmm import walk_forward_markov_filter

    rng = np.random.default_rng(4)
    t = 600
    x = rng.normal(1.0, 0.4, size=t)
    # Flicker spikes
    for i in range(300, 500, 3):
        x[i] = 10.0
    out = walk_forward_markov_filter(
        x, window=250, step=21, search_reps=2, maxiter=40
    )
    hard = out["hard"] >= 0
    if int(hard.sum()) < 20:
        pytest.skip("insufficient labeled days")
    d_hard = run_duration_stats(out["hard"] == 1)
    d_piger = run_duration_stats(out["hard_piger"] == 1)
    # Piger consecutive rule should not be shorter-run on average when both fire.
    assert d_piger["mean_turbulent_run_days"] >= d_hard["mean_turbulent_run_days"] - 1e-9
    assert "fit_hygiene" in out or "n_windows_attempted" in out
    assert out["n_windows_attempted"] >= 1
    assert 0.0 <= float(out["labeled_frac"]) <= 1.0


def test_markov_failed_fit_reuses_last_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Failed EM on window 2 must still label OOS via last causal params."""
    pytest.importorskip("statsmodels")
    import statsmodels.api as sm

    from mascotrl.eval.walk_forward_hmm import walk_forward_markov_filter

    x = _planted_turbulence_series(t=520, seed=7)
    t = x.shape[0]
    window, step = 250, 21

    real_fit = sm.tsa.MarkovRegression.fit
    seen_success = {"ok": False}

    def _flaky_fit(self, *args, **kwargs):
        if seen_success["ok"]:
            raise RuntimeError("planted_fit_failure")
        res = real_fit(self, *args, **kwargs)
        seen_success["ok"] = True
        return res

    monkeypatch.setattr(sm.tsa.MarkovRegression, "fit", _flaky_fit)
    out = walk_forward_markov_filter(
        x, window=window, step=step, search_reps=1, maxiter=40
    )
    # Second OOS block starts at window+step.
    second_block = window + step
    assert second_block < t
    assert out["hard"][second_block] >= 0, "expected reuse of last checkpoint"
    assert int(out["n_windows_reused"]) >= 1
    assert int(out["n_windows_failed"]) == 0 or int(out["n_windows_reused"]) >= 1
