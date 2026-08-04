"""Parity: purgedcv-backed CPCV vs hand-rolled AFML geometry (B-4).

Default: ``resolve_use_purgedcv`` -> purgedcv via ``src/eval/cpcv_lib.py``.
Hand-rolled ``src/eval/cpcv.py`` remains legacy when ``use_purgedcv: false``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("purgedcv")

from src.eval.cpcv import CPCVConfig, build_cpcv_folds, reconstruct_paths
from src.eval.cpcv_lib import build_cpcv_folds_lib, run_cpcv_lib


def _fold_train_test_index_sets(
    dates: list[pd.Timestamp], folds
) -> list[tuple[int, tuple[int, ...], frozenset[int], frozenset[int]]]:
    idx = {str(pd.Timestamp(d).date()): i for i, d in enumerate(dates)}
    out: list[tuple[int, tuple[int, ...], frozenset[int], frozenset[int]]] = []
    for f in folds:
        test_idx: set[int] = set()
        train_idx: set[int] = set()
        for w in f.test_windows:
            test_idx.update(range(idx[w["start"]], idx[w["end"]] + 1))
        for w in f.train_windows:
            train_idx.update(range(idx[w["start"]], idx[w["end"]] + 1))
        out.append((f.fold_id, f.test_groups, frozenset(test_idx), frozenset(train_idx)))
    return out


@pytest.fixture
def calendar() -> list[pd.Timestamp]:
    return list(pd.bdate_range("2018-01-01", periods=756))


@pytest.fixture
def residual_cfg() -> CPCVConfig:
    return CPCVConfig(n_splits=6, n_test_groups=2, purge_days=21, embargo_days=21)


def test_purgedcv_identical_train_test_index_sets_micro_case():
    """B-4: exact (train, test) index parity on 100-day micro geometry."""
    dates = list(pd.bdate_range("2020-01-01", periods=100))
    cfg = CPCVConfig(n_splits=5, n_test_groups=2, purge_days=2, embargo_days=2)
    legacy = _fold_train_test_index_sets(dates, build_cpcv_folds(dates, cfg))
    lib = _fold_train_test_index_sets(dates, build_cpcv_folds_lib(dates, cfg))
    assert len(legacy) == len(lib) == cfg.n_folds()
    for a, b in zip(legacy, lib):
        assert a[0] == b[0]
        assert a[1] == b[1]
        assert a[2] == b[2], f"test mismatch fold {a[1]}"
        assert a[3] == b[3], f"train mismatch fold {a[1]}"


def test_purgedcv_fold_boundaries_match_legacy(calendar, residual_cfg):
    legacy = build_cpcv_folds(calendar, residual_cfg)
    lib = build_cpcv_folds_lib(calendar, residual_cfg)
    assert len(legacy) == len(lib) == residual_cfg.n_folds()
    for a, b in zip(legacy, lib):
        assert a.fold_id == b.fold_id
        assert a.test_groups == b.test_groups
        assert a.test_windows == b.test_windows
        assert a.train_windows == b.train_windows
        assert a.n_test_days == b.n_test_days
        assert a.n_train_days == b.n_train_days
        assert abs(a.n_purged_days - b.n_purged_days) <= 1
        assert abs(a.n_embargoed_days - b.n_embargoed_days) <= 1


def test_purgedcv_path_shape_matches_legacy(calendar, residual_cfg):
    legacy = build_cpcv_folds(calendar, residual_cfg)
    lib = build_cpcv_folds_lib(calendar, residual_cfg)
    # Synthetic per-fold pnl covering all dates in each test window
    fold_pnl: dict[int, dict[str, float]] = {}
    for fold in legacy:
        pnl = {}
        for w in fold.test_windows:
            lo = calendar.index(pd.Timestamp(w["start"]))
            hi = calendar.index(pd.Timestamp(w["end"]))
            for i in range(lo, hi + 1):
                pnl[str(pd.Timestamp(calendar[i]).date())] = 0.001
        fold_pnl[fold.fold_id] = pnl
    paths_l = reconstruct_paths(calendar, legacy, fold_pnl, residual_cfg)
    paths_b = reconstruct_paths(calendar, lib, fold_pnl, residual_cfg)
    assert len(paths_l) == len(paths_b) == residual_cfg.n_paths()
    assert [p["n_days"] for p in paths_l] == [p["n_days"] for p in paths_b]


def test_purgedcv_extra_purge_indices(calendar, residual_cfg):
    extras = [100, 200, 300]
    legacy = build_cpcv_folds(
        calendar, residual_cfg, extra_purge_indices=extras, extra_purge_radius=5
    )
    lib = build_cpcv_folds_lib(
        calendar, residual_cfg, extra_purge_indices=extras, extra_purge_radius=5
    )
    for a, b in zip(legacy, lib):
        assert a.n_train_days == b.n_train_days
        assert a.train_windows == b.train_windows


def test_run_cpcv_lib_smoke(calendar, residual_cfg, tmp_path):
    def fold_runner(fold):
        out = {}
        for w in fold.test_windows:
            lo = calendar.index(pd.Timestamp(w["start"]))
            hi = calendar.index(pd.Timestamp(w["end"]))
            for i in range(lo, hi + 1):
                out[str(pd.Timestamp(calendar[i]).date())] = float(np.sin(i) * 0.01)
        return out

    art = run_cpcv_lib(
        calendar,
        fold_runner,
        residual_cfg,
        resume=True,
        out_dir=tmp_path,
        seed=0,
        arm="test",
    )
    assert art["backend"] == "purgedcv"
    assert art["n_folds"] == 15
    assert art["n_failed_folds"] == 0
    assert art["path_summary"]["n_paths"] == 5
    # Resume: second call should skip all folds
    art2 = run_cpcv_lib(
        calendar,
        fold_runner,
        residual_cfg,
        resume=True,
        out_dir=tmp_path,
        seed=0,
        arm="test",
    )
    assert art2["resume"]["n_skipped"] == 15
