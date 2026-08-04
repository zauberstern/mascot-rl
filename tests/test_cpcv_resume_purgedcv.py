"""Resume integrity for purgedcv CPCV backend."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("purgedcv")

from src.eval.cpcv import CPCVConfig, _CPCV_FOLD_AUX_KEY
from src.eval.cpcv_lib import run_cpcv_lib


def test_resume_restores_oos_aux_purgedcv(tmp_path: Path):
    dates = list(pd.bdate_range("2020-01-01", periods=180))
    cfg = CPCVConfig(n_splits=6, n_test_groups=2, purge_days=5, embargo_days=5)
    calls = {"n": 0}

    def fold_runner(fold):
        calls["n"] += 1
        pnl = {}
        for w in fold.test_windows:
            lo = dates.index(pd.Timestamp(w["start"]))
            hi = dates.index(pd.Timestamp(w["end"]))
            for i in range(lo, hi + 1):
                pnl[str(pd.Timestamp(dates[i]).date())] = 0.01
        pnl[_CPCV_FOLD_AUX_KEY] = {"weights": [[0.5, 0.5]], "fold": fold.fold_id}
        return pnl

    art1 = run_cpcv_lib(
        dates, fold_runner, cfg, resume=True, out_dir=tmp_path, seed=7, arm="eq"
    )
    n_first = calls["n"]
    assert n_first == art1["n_folds"]
    assert len(art1["fold_aux"]) == art1["n_folds"]

    art2 = run_cpcv_lib(
        dates, fold_runner, cfg, resume=True, out_dir=tmp_path, seed=7, arm="eq"
    )
    assert calls["n"] == n_first  # no re-runs
    assert art2["resume"]["n_skipped"] == art1["n_folds"]
    assert set(art2["fold_aux"]) == set(art1["fold_aux"])


def test_failed_fold_not_marked_complete_purgedcv(tmp_path: Path):
    dates = list(pd.bdate_range("2020-01-01", periods=180))
    cfg = CPCVConfig(n_splits=6, n_test_groups=2, purge_days=5, embargo_days=5)

    def fold_runner(fold):
        if fold.fold_id == 0:
            raise RuntimeError("boom")
        return {str(pd.Timestamp(dates[0]).date()): 0.0}

    art = run_cpcv_lib(
        dates, fold_runner, cfg, resume=True, out_dir=tmp_path, seed=0, arm="x"
    )
    assert 0 in art["failed_fold_ids"]
    # Retry: fold 0 still runs (not poisoned)
    runs = {"0": 0}

    def fold_runner2(fold):
        if fold.fold_id == 0:
            runs["0"] += 1
            return {str(pd.Timestamp(dates[i]).date()): 0.01 for i in range(10)}
        return {}

    art2 = run_cpcv_lib(
        dates, fold_runner2, cfg, resume=True, out_dir=tmp_path, seed=0, arm="x"
    )
    assert runs["0"] == 1
    assert art2["n_failed_folds"] == 0 or 0 not in art2["failed_fold_ids"]
