"""Part E.3: regime conditioning and Newey-West macro tilt sensitivity."""
from __future__ import annotations

import numpy as np
import pytest

from mascotrl.reporting import behavior_metrics as bm


def test_conditional_counts_sum_to_total_days():
    T, K = 30, 4
    W = np.full((T, K), 1.0 / K)
    S = np.eye(K, 7)
    regimes = np.array(
        ["calm"] * 10 + ["inflationary"] * 12 + ["crisis"] * 8, dtype=object
    )
    out = bm.regime_conditional_behaviour(W, sleeve_matrix=S, regimes=regimes)
    n_sum = sum(int(out[r]["n_days"]) for r in ("calm", "inflationary", "crisis"))
    assert n_sum == T


def test_empty_regime_yields_nan_and_zero_days():
    T, K = 12, 3
    W = np.full((T, K), 1.0 / K)
    S = np.eye(K, 7)
    regimes = np.array(["calm"] * T, dtype=object)
    out = bm.regime_conditional_behaviour(W, sleeve_matrix=S, regimes=regimes)
    assert out["crisis"]["n_days"] == 0
    assert out["inflationary"]["n_days"] == 0
    for mid in bm.BEHAVIOUR_MEASURE_IDS:
        assert mid in out["crisis"]
        assert np.isnan(out["crisis"][mid])


def test_macro_sensitivity_recovers_planted_beta():
    rng = np.random.default_rng(42)
    T, K = 500, 4
    S = np.zeros((K, 7))
    S[0, 3] = 1.0  # defensive name
    S[1, 0] = 1.0
    S[2, 1] = 1.0
    S[3, 2] = 1.0
    # Bounded macros so reconstructed weights stay in (0, 1)
    vix = np.clip(rng.standard_normal(T), -1.5, 1.5) * 0.4
    hy = rng.standard_normal(T) * 0.05
    term = rng.standard_normal(T) * 0.05
    b_true = 0.30
    tilt_def = np.zeros(T)
    tilt_def[1:] = (
        0.0
        + b_true * vix[:-1]
        + 0.02 * hy[:-1]
        + 0.02 * term[:-1]
        + rng.normal(0, 0.005, T - 1)
    )
    W = np.full((T, K), 0.25)
    w0 = tilt_def + 0.25
    assert float(w0.min()) > 0.05 and float(w0.max()) < 0.95
    W[:, 0] = w0
    rem = 1.0 - w0
    W[:, 1:] = (rem / 3.0)[:, None]
    realised = bm.sleeve_tilt_series(W, S)[:, 3]
    assert realised == pytest.approx(tilt_def, abs=1e-12)
    sens = bm.macro_tilt_sensitivity(
        W,
        sleeve_matrix=S,
        vix_z=vix,
        hy_oas_z=hy,
        term_spread=term,
        lags=21,
    )
    coef = sens["defensive"]["vix_z"]["coef"]
    assert coef == pytest.approx(b_true, abs=0.05)
    # Coefficient always paired with SE
    assert np.isfinite(sens["defensive"]["vix_z"]["se"])
    assert np.isfinite(sens["defensive"]["vix_z"]["tstat"])


def test_newey_west_se_exceeds_ols_on_autocorrelated_data():
    rng = np.random.default_rng(7)
    T = 400
    x = np.zeros(T)
    e = rng.standard_normal(T)
    for t in range(1, T):
        x[t] = 0.85 * x[t - 1] + rng.standard_normal()
    u = np.zeros(T)
    for t in range(1, T):
        u[t] = 0.9 * u[t - 1] + e[t]
    y = 0.5 + 0.4 * x + u
    X = x.reshape(-1, 1)
    nw = bm.newey_west_ols(y, X, lags=21)
    ols = bm.ols_with_se(y, X)
    assert nw["se"][1] > ols["se"][1]
    assert nw["se"][0] > ols["se"][0]


def test_turbulence_overlay_preserves_inflationary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Option (a): turbulent days promote calm→crisis only; inflationary stays."""
    t, n = 12, 3
    returns = np.zeros((t, n), dtype=np.float64)
    existing = np.array(
        ["calm", "inflationary", "calm", "crisis", "inflationary"] + ["calm"] * 7,
        dtype=object,
    )
    turb_flags = np.array(
        [True, True, True, True, False] + [False] * 7, dtype=bool
    )
    out = bm.turbulence_regimes_from_returns(
        returns, existing=existing, crisis_mask=turb_flags
    )
    assert out is not None
    assert out[0] == "crisis"  # calm + turb
    assert out[1] == "inflationary"  # preserved
    assert out[2] == "crisis"
    assert out[3] == "crisis"  # already crisis
    assert out[4] == "inflationary"


def test_overlay_markov_ignores_isolated_q75_spikes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default overlay_mode=markov: q75 flicker alone must not promote calm→crisis."""
    t, n = 30, 3
    returns = np.zeros((t, n), dtype=np.float64)
    existing = np.array(["calm"] * t, dtype=object)
    q75_spikes = np.zeros(t, dtype=bool)
    q75_spikes[[5, 8, 12, 15, 20]] = True

    monkeypatch.setattr(
        "mascotrl.eval.turbulence.turbulence_index",
        lambda r, **kwargs: np.ones(t),
    )
    monkeypatch.setattr(
        "mascotrl.eval.turbulence.classify_regime",
        lambda turb, **kwargs: q75_spikes,
    )

    def _calm_markov(series, **kwargs):
        hard = np.zeros(t, dtype=np.int32)  # all calm (hard==0)
        return {
            "hard": hard,
            "p_highvol": np.zeros(t),
            "hard_piger": np.zeros(t, dtype=np.int32),
            "train_ends": [],
        }

    monkeypatch.setattr(
        "mascotrl.eval.walk_forward_hmm.walk_forward_markov_filter",
        _calm_markov,
    )
    out = bm.turbulence_regimes_from_returns(
        returns, existing=existing, overlay_mode="markov", hmm_window=10, hmm_step=5
    )
    assert out is not None
    assert all(str(x) == "calm" for x in out)


def test_overlay_q75_mode_still_promotes(monkeypatch: pytest.MonkeyPatch) -> None:
    t, n = 12, 3
    returns = np.zeros((t, n), dtype=np.float64)
    existing = np.array(["calm"] * t, dtype=object)
    turb_flags = np.array([True, False] + [False] * 10, dtype=bool)
    monkeypatch.setattr(
        "mascotrl.eval.turbulence.turbulence_index",
        lambda r, **kwargs: np.ones(t),
    )
    monkeypatch.setattr(
        "mascotrl.eval.turbulence.classify_regime",
        lambda turb, **kwargs: turb_flags,
    )
    out = bm.turbulence_regimes_from_returns(
        returns, existing=existing, overlay_mode="q75"
    )
    assert out is not None
    assert out[0] == "crisis"
    assert out[1] == "calm"
