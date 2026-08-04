"""CRUCIBLE behavioural sleeves: scores, primary, quotas, fill order."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.conftest import FLOAT_TOL

from src.data.crucible import (
    SLEEVE_FILL_ORDER,
    SLEEVE_IDS,
    SLEEVE_QUOTAS,
    assign_sleeves,
    sleeve_scores,
)


def _toy_panels(n: int = 40, t: int = 280, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=t)
    secids = list(range(1000, 1000 + n))
    resid = pd.DataFrame(rng.normal(0, 0.01, size=(t, n)), index=dates, columns=secids)
    # Inject sleeve-discriminative structure
    resid.iloc[-252:-21, :8] += 0.002  # trend winners
    resid.iloc[-21:, 8:16] -= 0.003  # recent losers -> reversal
    adv = pd.DataFrame(
        rng.uniform(1e6, 5e7, size=(t, n)), index=dates, columns=secids
    )
    beta = pd.DataFrame(
        rng.uniform(0.4, 1.6, size=(t, n)), index=dates, columns=secids
    )
    rows = []
    for sid in secids:
        for d in dates[-80:]:
            for sig, val in (
                ("mfis_30", float(rng.normal())),
                ("mfis_365", float(rng.normal())),
                ("hv_20", float(rng.uniform(0.1, 0.4))),
                ("iv_30", float(rng.uniform(0.1, 0.5))),
                ("skew", float(rng.normal(0, 0.2))),
                ("term", float(rng.normal(0, 0.1))),
            ):
                rows.append({"date": d, "secid": sid, "signal": sig, "value": val})
    surface = pd.DataFrame(rows)
    return resid, surface, adv, beta


def test_sleeve_scores_finite_and_zscored():
    resid, surface, adv, beta = _toy_panels()
    scores = sleeve_scores(resid, surface, adv, beta)
    for col in ("trend", "reversal", "carry", "defensive", "lottery", "illiquid"):
        assert col in scores.columns
        s = scores[col].to_numpy(dtype=float)
        assert np.isfinite(s).all()
        assert abs(float(np.mean(s))) < 0.25
        assert 0.5 < float(np.std(s)) < 1.5


def test_sleeve_primary_is_argmax_of_z_matrix():
    resid, surface, adv, beta = _toy_panels()
    scores = sleeve_scores(resid, surface, adv, beta)
    community_of = {int(s): int(i % 8) for i, s in enumerate(scores.index)}
    quotas = {k: max(1, v // 10) for k, v in SLEEVE_QUOTAS.items()}
    membership, primary = assign_sleeves(
        scores, quotas=quotas, community_of=community_of, max_per_community=5
    )
    assert set(primary.keys()) == set(int(s) for s in scores.index if int(s) in primary)
    style_cols = [c for c in SLEEVE_IDS if c != "core"]
    for sid, sleeve in primary.items():
        if sleeve == "core":
            continue
        row = scores.loc[sid, style_cols]
        assert sleeve == str(row.idxmax())


def test_sleeve_matrix_column_order_equals_sleeve_ids():
    from src.data.crucible import build_sleeve_matrix

    primary = {1001: "trend", 1002: "core", 1003: "lottery"}
    membership = {
        "trend": [1001],
        "lottery": [1003, 1001],
        "core": [1002],
        "reversal": [],
        "carry": [],
        "defensive": [],
        "illiquid": [],
    }
    secids = [1001, 1002, 1003]
    mat = build_sleeve_matrix(secids, membership)
    assert mat.shape == (3, len(SLEEVE_IDS))
    assert list(SLEEVE_IDS) == [
        "trend",
        "reversal",
        "carry",
        "defensive",
        "lottery",
        "illiquid",
        "core",
    ]
    assert mat.dtype == np.float32
    assert mat[0, SLEEVE_IDS.index("trend")] == pytest.approx(1.0, **FLOAT_TOL)
    assert mat[0, SLEEVE_IDS.index("lottery")] == pytest.approx(1.0, **FLOAT_TOL)


def test_quotas_met_when_pool_rich():
    resid, surface, adv, beta = _toy_panels(n=60)
    scores = sleeve_scores(resid, surface, adv, beta)
    community_of = {int(s): int(i % 12) for i, s in enumerate(scores.index)}
    quotas = {
        "trend": 3,
        "reversal": 3,
        "carry": 3,
        "defensive": 2,
        "lottery": 2,
        "illiquid": 2,
        "core": 2,
    }
    membership, primary = assign_sleeves(
        scores,
        quotas=quotas,
        community_of=community_of,
        max_per_community=4,
    )
    for sleeve, q in quotas.items():
        n_primary = sum(1 for s in primary.values() if s == sleeve)
        assert n_primary == q, f"{sleeve}: got {n_primary} want {q}"


def test_starved_sleeve_records_shortfall_without_breaking_community_cap():
    resid, surface, adv, beta = _toy_panels(n=20)
    scores = sleeve_scores(resid, surface, adv, beta)
    # Force all names into one community so community cap binds hard
    community_of = {int(s): 0 for s in scores.index}
    quotas = {
        "trend": 5,
        "reversal": 5,
        "carry": 5,
        "defensive": 5,
        "lottery": 5,
        "illiquid": 5,
        "core": 5,
    }
    membership, primary, shortfalls = assign_sleeves(
        scores,
        quotas=quotas,
        community_of=community_of,
        max_per_community=3,
        return_shortfalls=True,
    )
    assert any(v > 0 for v in shortfalls.values())
    from collections import Counter

    counts = Counter(community_of[s] for s in primary)
    assert counts[0] <= 3 * len([s for s in SLEEVE_IDS])  # soft
    # Hard: selected names never exceed max_per_community in one community
    # (with one community, total selected <= 3)
    assert len(primary) <= 3


def test_greedy_fill_order_is_scarce_first():
    assert SLEEVE_FILL_ORDER == (
        "illiquid",
        "lottery",
        "defensive",
        "carry",
        "reversal",
        "trend",
        "core",
    )
