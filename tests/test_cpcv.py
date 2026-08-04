"""Locking tests for Combinatorial Purged Cross-Validation (W4).

Protocol per Lopez de Prado, Advances in Financial Machine Learning (2018),
chapters 7 and 12.
"""
from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
import pytest

from src.eval.cpcv import (
    CPCVConfig,
    assign_paths,
    build_cpcv_folds,
    group_bounds,
    reconstruct_paths,
    run_cpcv,
    summarize_paths,
)


@pytest.fixture
def dates():
    return list(pd.bdate_range("2015-01-01", periods=600))


def test_cpcv_config_defaults_are_purge_embargo_21() -> None:
    cfg = CPCVConfig()
    assert cfg.purge_days == 21
    assert cfg.embargo_days == 21


def test_default_geometry_is_afml_worked_example():
    cfg = CPCVConfig()
    assert (cfg.n_splits, cfg.n_test_groups) == (6, 2)
    assert cfg.n_folds() == comb(6, 2) == 15
    assert cfg.n_paths() == comb(5, 1) == 5


def test_config_validation_rejects_bad_geometry():
    with pytest.raises(ValueError):
        CPCVConfig(n_splits=1).validate()
    with pytest.raises(ValueError):
        CPCVConfig(n_splits=6, n_test_groups=6).validate()
    with pytest.raises(ValueError):
        CPCVConfig(purge_days=-1).validate()


def test_group_bounds_partition_without_gaps_or_overlap(dates):
    bounds = group_bounds(dates, 6)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == len(dates) - 1
    for (lo1, hi1), (lo2, _) in zip(bounds, bounds[1:]):
        assert hi1 + 1 == lo2


def test_fold_count_and_test_coverage(dates):
    folds = build_cpcv_folds(dates, CPCVConfig())
    assert len(folds) == 15
    # Every group is tested in exactly C(n-1, k-1) folds.
    counts = {g: 0 for g in range(6)}
    for f in folds:
        for g in f.test_groups:
            counts[g] += 1
    assert set(counts.values()) == {comb(5, 1)}


def test_train_and_test_are_disjoint_with_purge_and_embargo(dates):
    cfg = CPCVConfig(purge_days=3, embargo_days=4)
    folds = build_cpcv_folds(dates, cfg)
    idx = {str(pd.Timestamp(d).date()): i for i, d in enumerate(dates)}
    for f in folds:
        test_idx = set()
        for w in f.test_windows:
            test_idx.update(range(idx[w["start"]], idx[w["end"]] + 1))
        train_idx = set()
        for w in f.train_windows:
            train_idx.update(range(idx[w["start"]], idx[w["end"]] + 1))
        assert not (test_idx & train_idx), f"fold {f.fold_id} train/test overlap"
        # No training day may sit inside the purge or embargo buffer.
        for i in sorted(test_idx):
            for j in range(i - cfg.purge_days, i):
                if j >= 0 and j not in test_idx:
                    assert j not in train_idx
            for j in range(i + 1, i + 1 + cfg.embargo_days):
                if j < len(dates) and j not in test_idx:
                    assert j not in train_idx


def test_purge_and_embargo_actually_remove_days(dates):
    cfg = CPCVConfig(purge_days=5, embargo_days=5)
    strict = build_cpcv_folds(dates, cfg)
    loose = build_cpcv_folds(dates, CPCVConfig(purge_days=0, embargo_days=0))
    assert sum(f.n_train_days for f in strict) < sum(f.n_train_days for f in loose)
    n_groups = cfg.n_splits
    for f in strict:
        # Purging is only possible where a non-test group precedes a test block;
        # the fold testing the leading groups has nothing before it to purge.
        can_purge = any(g > 0 and (g - 1) not in f.test_groups for g in f.test_groups)
        can_embargo = any(
            g < n_groups - 1 and (g + 1) not in f.test_groups for g in f.test_groups
        )
        if can_purge:
            assert f.n_purged_days > 0, f"fold {f.fold_id} should purge"
        if can_embargo:
            assert f.n_embargoed_days > 0, f"fold {f.fold_id} should embargo"
    assert sum(f.n_purged_days for f in strict) > 0
    assert sum(f.n_embargoed_days for f in strict) > 0


def test_paths_cover_every_group_exactly_once():
    cfg = CPCVConfig()
    paths = assign_paths(cfg)
    assert len(paths) == cfg.n_paths()
    for path in paths:
        assert [g for g, _ in path] == list(range(cfg.n_splits))
    # Each fold is consumed the right number of times overall.
    used: dict[int, int] = {}
    for path in paths:
        for _, fid in path:
            used[fid] = used.get(fid, 0) + 1
    assert sum(used.values()) == cfg.n_splits * cfg.n_paths()


def test_reconstructed_paths_span_the_sample(dates):
    cfg = CPCVConfig()
    folds = build_cpcv_folds(dates, cfg)
    rng = np.random.default_rng(0)
    fold_pnl = {
        f.fold_id: {
            str(pd.Timestamp(d).date()): float(rng.standard_normal() * 0.01)
            for w in f.test_windows
            for d in pd.bdate_range(w["start"], w["end"])
            if str(pd.Timestamp(d).date())
            in {str(pd.Timestamp(x).date()) for x in dates}
        }
        for f in folds
    }
    paths = reconstruct_paths(dates, folds, fold_pnl, cfg)
    assert len(paths) == cfg.n_paths()
    for p in paths:
        # A path is a strictly increasing, non-overlapping walk.
        assert p["dates"] == sorted(p["dates"])
        assert len(p["dates"]) == len(set(p["dates"]))
        assert p["n_days"] > 0


def test_summarize_paths_reports_a_distribution():
    paths = [{"sharpe": s} for s in (0.5, 1.0, -0.2, 0.8, 0.1)]
    s = summarize_paths(paths)
    assert s["n_paths"] == 5
    assert s["sharpe_mean"] == pytest.approx(0.44)
    assert s["positive_path_rate"] == pytest.approx(0.8)
    assert s["sharpe_p05"] <= s["sharpe_median"] <= s["sharpe_p95"]


def test_summarize_paths_handles_all_nan():
    s = summarize_paths([{"sharpe": float("nan")}])
    assert np.isnan(s["sharpe_mean"])
    assert s["n_paths"] == 1


def test_run_cpcv_produces_path_distribution_not_single_number(dates):
    cfg = CPCVConfig(n_splits=5, n_test_groups=2)

    def runner(fold):
        rng = np.random.default_rng(fold.fold_id)
        out = {}
        for w in fold.test_windows:
            for d in pd.bdate_range(w["start"], w["end"]):
                out[str(pd.Timestamp(d).date())] = float(rng.standard_normal() * 0.01)
        return out

    res = run_cpcv(dates, runner, cfg)
    assert res["protocol"] == "combinatorial_purged_cv"
    assert res["n_folds"] == comb(5, 2)
    assert len(res["paths"]) == comb(4, 1)
    assert len(res["path_summary"]["path_sharpes"]) == comb(4, 1)
    assert "does not correct meta-overfitting" in res["scope_note"]


def test_run_cpcv_survives_a_failing_fold(dates):
    cfg = CPCVConfig(n_splits=4, n_test_groups=2)

    def runner(fold):
        if fold.fold_id == 0:
            raise RuntimeError("simulated fold failure")
        return {
            str(pd.Timestamp(d).date()): 0.001
            for w in fold.test_windows
            for d in pd.bdate_range(w["start"], w["end"])
        }

    res = run_cpcv(dates, runner, cfg)
    assert res["n_folds"] == comb(4, 2)
    assert res["folds"][0]["n_pnl_days"] == 0
