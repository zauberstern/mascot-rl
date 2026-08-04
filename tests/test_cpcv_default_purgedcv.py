"""Default CPCV backend is purgedcv when installed."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("purgedcv")

from src.eval.cpcv import CPCVConfig, _CPCV_FOLD_AUX_KEY
from src.eval.cpcv_backend import resolve_use_purgedcv
from src.eval.cpcv_lib import build_cpcv_folds_lib, run_cpcv_lib


def test_resolve_use_purgedcv_defaults_true():
    assert resolve_use_purgedcv({}) is True
    assert resolve_use_purgedcv({"use_purgedcv": False}) is False
    assert resolve_use_purgedcv({"use_purgedcv": True}) is True


def test_lib_and_legacy_fold_ids_match():
    from src.eval.cpcv import build_cpcv_folds

    dates = list(pd.bdate_range("2020-01-01", periods=120))
    cfg = CPCVConfig(n_splits=5, n_test_groups=2, purge_days=3, embargo_days=2)
    lib_ids = [f.fold_id for f in build_cpcv_folds_lib(dates, cfg)]
    legacy_ids = [f.fold_id for f in build_cpcv_folds(dates, cfg)]
    assert lib_ids == legacy_ids
    assert len(lib_ids) == cfg.n_folds()


def test_resume_fold_ids_stable(tmp_path: Path):
    dates = list(pd.bdate_range("2020-01-01", periods=120))
    cfg = CPCVConfig(n_splits=5, n_test_groups=2, purge_days=3, embargo_days=2)
    folds_a = build_cpcv_folds_lib(dates, cfg)
    folds_b = build_cpcv_folds_lib(dates, cfg)
    assert [f.fold_id for f in folds_a] == [f.fold_id for f in folds_b]

    calls = {"n": 0}

    def fold_runner(fold):
        calls["n"] += 1
        pnl = {}
        for w in fold.test_windows:
            lo = dates.index(pd.Timestamp(w["start"]))
            hi = dates.index(pd.Timestamp(w["end"]))
            for i in range(lo, hi + 1):
                pnl[str(pd.Timestamp(dates[i]).date())] = 0.01
        pnl[_CPCV_FOLD_AUX_KEY] = {"fold": fold.fold_id}
        return pnl

    art1 = run_cpcv_lib(
        dates, fold_runner, cfg, resume=True, out_dir=tmp_path, seed=3, arm="eq"
    )
    n_folds = int(art1["n_folds"])
    assert calls["n"] == n_folds
    art2 = run_cpcv_lib(
        dates, fold_runner, cfg, resume=True, out_dir=tmp_path, seed=3, arm="eq"
    )
    assert int(art2["n_folds"]) == n_folds
    assert calls["n"] == n_folds  # second run fully resumed
    assert len(art1.get("fold_aux") or []) == n_folds
    assert len(art2.get("fold_aux") or []) == n_folds
