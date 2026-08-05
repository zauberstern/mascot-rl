"""B-3 / E-4: hand-computed CPCV micro-case on a 100-day index.

AFML delta (fixed trading-day purge vs per-sample ``t1``): OK-with-rationale
because delta-hedged labels span one session; ``purge_days=2`` is conservative.
See ``logs/campaign_sprint/AUDIT_LEDGER.md`` [B-3].
"""
from __future__ import annotations

from math import comb

import pandas as pd
import pytest

from mascotrl.eval.cpcv import CPCVConfig, assign_paths, build_cpcv_folds, group_bounds


def _fold_index_sets(
    dates: list[pd.Timestamp], folds
) -> dict[tuple[int, ...], tuple[frozenset[int], frozenset[int]]]:
    idx = {str(pd.Timestamp(d).date()): i for i, d in enumerate(dates)}
    out: dict[tuple[int, ...], tuple[frozenset[int], frozenset[int]]] = {}
    for f in folds:
        test_idx: set[int] = set()
        train_idx: set[int] = set()
        for w in f.test_windows:
            test_idx.update(range(idx[w["start"]], idx[w["end"]] + 1))
        for w in f.train_windows:
            train_idx.update(range(idx[w["start"]], idx[w["end"]] + 1))
        out[f.test_groups] = (frozenset(test_idx), frozenset(train_idx))
    return out


@pytest.fixture
def micro_dates() -> list[pd.Timestamp]:
    return list(pd.bdate_range("2020-01-01", periods=100))


@pytest.fixture
def micro_cfg() -> CPCVConfig:
    return CPCVConfig(
        n_splits=5, n_test_groups=2, purge_days=2, embargo_days=2
    )


def test_cpcv_micro_case_fold_and_path_counts(micro_dates, micro_cfg):
    folds = build_cpcv_folds(micro_dates, micro_cfg)
    assert len(folds) == comb(5, 2) == 10
    paths = assign_paths(micro_cfg)
    assert len(paths) == comb(4, 1) == 4


def test_cpcv_micro_case_group_bounds(micro_dates, micro_cfg):
    bounds = group_bounds(micro_dates, micro_cfg.n_splits)
    assert bounds == [(0, 19), (20, 39), (40, 59), (60, 79), (80, 99)]


def test_cpcv_micro_case_fold_01_hand_computed_indices(micro_dates, micro_cfg):
    """Fold testing groups (0, 1): fold_id=0.

    Hand computation (100 days, 5 equal groups of 20, purge=2, embargo=2):
    - test groups 0+1 -> indices 0..39
    - embargo after group 1 (hi=39): indices 40, 41 (within 2 days after test block)
    - no purge before group 0 (lo=0); purge before group 1 hits 18-19 already in test
    - train = 42..99 (58 days)
    """
    folds = build_cpcv_folds(micro_dates, micro_cfg)
    by_group = _fold_index_sets(micro_dates, folds)
    test_idx, train_idx = by_group[(0, 1)]

    assert test_idx == frozenset(range(40))
    assert train_idx == frozenset(range(42, 100))
    assert len(test_idx) == 40
    assert len(train_idx) == 58
    assert not (test_idx & train_idx)

    fold0 = next(f for f in folds if f.test_groups == (0, 1))
    assert fold0.fold_id == 0
    assert fold0.n_purged_days == 0
    assert fold0.n_embargoed_days == 2


def test_cpcv_micro_case_fold_34_hand_computed_indices(micro_dates, micro_cfg):
    """Fold testing groups (3, 4): fold_id=9.

    - test -> 60..99
    - purge before group 3 (lo=60): indices 58, 59
    - train -> 0..57 (58 days)
    """
    folds = build_cpcv_folds(micro_dates, micro_cfg)
    by_group = _fold_index_sets(micro_dates, folds)
    test_idx, train_idx = by_group[(3, 4)]

    assert test_idx == frozenset(range(60, 100))
    assert train_idx == frozenset(range(58))
    assert len(test_idx) == 40
    assert len(train_idx) == 58

    fold9 = next(f for f in folds if f.test_groups == (3, 4))
    assert fold9.fold_id == 9
    assert fold9.n_purged_days == 2
    assert fold9.n_embargoed_days == 0
