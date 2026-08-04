"""Part E.2: behaviour vector measures against hand-computed fixtures."""
from __future__ import annotations

import math
import re

import numpy as np
import pytest

from src.reporting import behavior_metrics as bm


def _fixture_5x4():
    """T=5, K=4 weight path and 7-col primary sleeve partition."""
    W = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0, 1.0],
            [0.1, 0.2, 0.3, 0.4],
        ],
        dtype=np.float64,
    )
    S = np.zeros((4, 7), dtype=np.float64)
    # primary partition: one name per first four sleeves
    S[0, 0] = 1.0  # trend
    S[1, 1] = 1.0  # reversal
    S[2, 2] = 1.0  # carry
    S[3, 3] = 1.0  # defensive
    R = np.array(
        [
            [0.01, 0.00, -0.01, 0.02],
            [0.02, -0.01, 0.00, 0.01],
            [-0.02, 0.01, 0.01, -0.01],
            [0.00, 0.02, -0.02, 0.03],
            [0.01, 0.01, 0.01, -0.02],
        ],
        dtype=np.float64,
    )
    return W, S, R


def _hand_hhi(W: np.ndarray) -> float:
    return float(np.mean(np.sum(W * W, axis=1)))


def _hand_turnover(W: np.ndarray) -> float:
    d = np.abs(np.diff(W, axis=0)).sum(axis=1) * 0.5
    return float(np.mean(d))


def _hand_entropy(W: np.ndarray) -> float:
    ents = []
    for row in W:
        e = 0.0
        for w in row:
            if w > 0.0:
                e -= float(w) * math.log(float(w))
        ents.append(e)
    return float(np.mean(ents))


def test_twenty_three_measures_match_hand_computed():
    W, S, R = _fixture_5x4()
    out = bm.compute_behaviour_vector(
        W, asset_returns=R, sleeve_matrix=S, turnover_cap=0.4
    )
    assert set(out) >= set(bm.BEHAVIOUR_MEASURE_IDS)
    assert len(bm.BEHAVIOUR_MEASURE_IDS) == 43
    assert len(bm.COMPOSITION_MEASURE_IDS) == 34
    assert set(bm.COMPOSITION_MEASURE_IDS).issubset(set(bm.BEHAVIOUR_MEASURE_IDS))
    assert not {
        "semantic_rotation_rate",
        "support_jaccard_lag1",
        "style_agreement_cosine",
    } & set(bm.COMPOSITION_MEASURE_IDS)

    hhi = _hand_hhi(W)
    assert out["hhi_mean"] == pytest.approx(hhi, abs=1e-12)
    assert out["n_eff_mean"] == pytest.approx(float(np.mean(1.0 / np.sum(W * W, axis=1))), abs=1e-12)
    assert out["max_weight_mean"] == pytest.approx(float(np.mean(W.max(axis=1))), abs=1e-12)
    ew = np.full(4, 0.25)
    l1 = float(np.mean(np.abs(W - ew).sum(axis=1)))
    assert out["l1_vs_ew_mean"] == pytest.approx(l1, abs=1e-12)
    to = _hand_turnover(W)
    assert out["turnover_mean"] == pytest.approx(to, abs=1e-12)
    # binding: steps 1..4 where 0.5*L1 >= 0.4
    step_to = 0.5 * np.abs(np.diff(W, axis=0)).sum(axis=1)
    assert out["turnover_cap_binding_frac"] == pytest.approx(
        float(np.mean(step_to >= 0.4 - 1e-15)), abs=1e-12
    )
    assert out["action_entropy_mean"] == pytest.approx(_hand_entropy(W), abs=1e-12)
    assert out["holding_period_days"] == pytest.approx(1.0 / to, abs=1e-12)

    # tilts at t=0: w=[1,0,0,0], n_s/K = 0.25 for first four
    tilts = bm.sleeve_tilt_series(W, S)
    assert tilts.shape == (5, 7)
    assert tilts[0, 0] == pytest.approx(1.0 - 0.25, abs=1e-12)
    assert tilts[0, 1] == pytest.approx(0.0 - 0.25, abs=1e-12)
    for j, sid in enumerate(bm.SLEEVE_IDS):
        assert out[f"tilt_{sid}"] == pytest.approx(float(np.mean(tilts[:, j])), abs=1e-12)

    # risk shape hand checks
    r_p = np.sum(W * R, axis=1)
    r_ew = R.mean(axis=1)
    down = r_ew < 0
    up = r_ew > 0
    assert out["downside_capture"] == pytest.approx(
        float(r_p[down].sum() / r_ew[down].sum()), abs=1e-12
    )
    assert out["upside_capture"] == pytest.approx(
        float(r_p[up].sum() / r_ew[up].sum()), abs=1e-12
    )
    assert out["return_skew"] == pytest.approx(
        float(
            ((r_p - r_p.mean()) ** 3).mean()
            / (r_p.std(ddof=0) ** 3 + 1e-24)
        ),
        abs=1e-10,
    )
    # max drawdown of wealth path
    wealth = np.cumprod(1.0 + r_p)
    peak = np.maximum.accumulate(wealth)
    dd = 1.0 - wealth / peak
    assert out["max_drawdown"] == pytest.approx(float(dd.max()), abs=1e-12)
    q = np.quantile(r_p, 0.05)
    assert out["cvar_05"] == pytest.approx(float(r_p[r_p <= q].mean()), abs=1e-12)


def test_primary_tilts_sum_to_zero():
    W, S, _ = _fixture_5x4()
    tilts = bm.sleeve_tilt_series(W, S)
    # primary partition covers all names exactly once
    assert np.allclose(tilts.sum(axis=1), 0.0, atol=1e-12)


def test_equal_weight_path_zero_active_share_and_primary_tilts():
    k = 4
    W = np.full((6, k), 1.0 / k)
    S = np.zeros((k, 7))
    for i in range(k):
        S[i, i] = 1.0
    out = bm.compute_behaviour_vector(W, sleeve_matrix=S)
    assert out["l1_vs_ew_mean"] == pytest.approx(0.0, abs=1e-12)
    for sid in bm.SLEEVE_IDS[:4]:
        assert out[f"tilt_{sid}"] == pytest.approx(0.0, abs=1e-12)


def test_single_name_path_hhi_and_neff():
    W = np.zeros((8, 5))
    W[:, 0] = 1.0
    out = bm.compute_behaviour_vector(W)
    assert out["hhi_mean"] == pytest.approx(1.0, abs=1e-12)
    assert out["n_eff_mean"] == pytest.approx(1.0, abs=1e-12)


def test_every_measure_docstring_states_unit():
    # Core path measures live as measure_* in this module. Holdings / RBSA /
    # regime-delta extensions are filled by composition layers elsewhere.
    layered = {
        "exposure_size",
        "exposure_value",
        "exposure_momentum",
        "exposure_quality",
        "exposure_low_vol",
        "sector_hhi",
        "rbsa_r_squared",
        "delta_hhi_regime",
        "delta_turnover_regime",
        "delta_defensive_regime",
        "delta_quality_regime",
        "semantic_rotation_rate",
        "semantic_pc1_mean",
        "semantic_pc2_mean",
        "semantic_pc3_mean",
        "style_agreement_cosine",
    }
    for name in bm.BEHAVIOUR_MEASURE_IDS:
        if name in layered:
            continue
        fn = getattr(bm, f"measure_{name}", None)
        assert fn is not None, f"missing measure_{name}"
        doc = fn.__doc__ or ""
        assert re.search(r"(?i)\bunit\b", doc), f"{name} docstring missing unit"
