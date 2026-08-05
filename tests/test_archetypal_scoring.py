"""Composition-based archetype scoring (AA / NMF / GMM)."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import FLOAT_TOL


def test_fit_composition_rows_sum_to_one_aa_and_fallbacks():
    from mascotrl.reporting.archetypal_scoring import fit_composition

    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 8))
    x = (x - x.mean(0)) / (x.std(0) + 1e-8)
    for method in ("aa", "nmf", "gmm"):
        alpha, archetypes = fit_composition(x, k=5, method=method)
        assert alpha.shape == (20, 5)
        assert archetypes.shape[0] == 5
        assert np.allclose(alpha.sum(axis=1), 1.0, atol=1e-5)
        assert (alpha >= -1e-8).all()


def test_no_mixed_string_in_archetypal_scoring_source():
    src = Path("src/mascotrl/reporting/archetypal_scoring.py").read_text(encoding="utf-8")
    assert "mixed" not in src.lower()


def test_name_archetypes_extreme_panel():
    from mascotrl.reporting.archetypal_scoring import name_archetypes
    from mascotrl.reporting.policy_behavior import ARCHETYPE_SCORE_WEIGHTS

    # Build synthetic archetype rows that match seed weight directions.
    feature_names = sorted(
        {k for w in ARCHETYPE_SCORE_WEIGHTS.values() for k in w}
    )
    Z = np.zeros((len(ARCHETYPE_SCORE_WEIGHTS), len(feature_names)))
    names_expected = list(ARCHETYPE_SCORE_WEIGHTS.keys())
    for i, arch in enumerate(names_expected):
        for j, feat in enumerate(feature_names):
            Z[i, j] = float(ARCHETYPE_SCORE_WEIGHTS[arch].get(feat, 0.0))
    named = name_archetypes(Z, feature_names=feature_names)
    assert named == names_expected


def test_choose_k_returns_finite_table():
    from mascotrl.reporting.archetypal_scoring import choose_k

    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 6))
    x = (x - x.mean(0)) / (x.std(0) + 1e-8)
    table = choose_k(x, ks=range(3, 6))
    assert set(table.keys()) >= {3, 4, 5}
    for k, row in table.items():
        assert np.isfinite(row.get("rss", np.nan)) or np.isfinite(row.get("bic", np.nan))


def test_stamp_composition_fields():
    from mascotrl.reporting.archetypal_scoring import composition_for_rows

    rng = np.random.default_rng(2)
    feature_names = [
        "hhi_mean",
        "turnover_mean",
        "tilt_trend",
        "tilt_reversal",
        "tilt_defensive",
        "tilt_lottery",
        "rotation_rate",
        "exposure_size",
    ]
    rows = [
        {f: float(rng.normal()) for f in feature_names}
        for _ in range(12)
    ]
    stamped = composition_for_rows(rows, feature_names=feature_names, k=5)
    assert len(stamped) == 12
    for item in stamped:
        comp = item["archetype_composition"]
        assert abs(sum(comp.values()) - 1.0) < 1e-5
        assert item["archetype_primary"] == max(comp, key=comp.get)
        assert item["archetype_confidence"] == pytest.approx(
            max(comp.values()), **FLOAT_TOL
        )
        assert "mixed" not in comp


def test_select_k_from_table_prefers_min_bic():
    from mascotrl.reporting.archetypal_scoring import select_k_from_table

    table = {
        3: {"rss": 10.0, "bic": 100.0},
        4: {"rss": 8.0, "bic": 50.0},
        5: {"rss": 7.0, "bic": 80.0},
    }
    assert select_k_from_table(table) == 4


def test_select_k_from_table_fallback_locked_five():
    from mascotrl.reporting.archetypal_scoring import select_k_from_table

    table = {
        3: {"rss": 10.0, "bic": float("nan")},
        4: {"rss": 9.0, "bic": float("nan")},
        5: {"rss": 8.0, "bic": float("nan")},
    }
    assert select_k_from_table(table) == 5


def test_bootstrap_ari_high_on_separated_blobs():
    from mascotrl.reporting.archetypal_scoring import bootstrap_ari

    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.0, scale=0.2, size=(10, 2))
    b = rng.normal(loc=5.0, scale=0.2, size=(10, 2))
    x = np.vstack([a, b])
    out = bootstrap_ari(x, k=2, n_boot=10, frac=0.8, seed=0, method="gmm")
    assert out["status"] == "ok"
    assert np.isfinite(out["ari_mean"])
    assert float(out["ari_mean"]) > 0.3


def test_bootstrap_ari_too_small_panel():
    from mascotrl.reporting.archetypal_scoring import bootstrap_ari

    x = np.random.default_rng(0).normal(size=(4, 3))
    out = bootstrap_ari(x, k=2)
    assert out["status"] == "skipped"
    assert out["reason"] == "too_few_cells"
