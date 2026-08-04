"""CPCV campaign manifest resume: skip completed (fold, seed, arm) cells."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.plumbing
from tests.conftest import FLOAT_TOL

import src.eval.research_alpha_cpcv as research_alpha_cpcv
from src.eval.campaign_manifest import (
    cell_key,
    is_cell_complete,
    load_manifest,
    mark_cell_complete,
    purge_orphan_fold_cells,
    save_manifest,
)
from src.eval.cpcv import CPCVConfig, CPCVFold, run_cpcv


def test_cell_key_stable() -> None:
    assert cell_key(0, 42, "opt") == "0|42|opt"
    assert cell_key(1, 0, "eq") == "1|0|eq"


def test_manifest_roundtrip_atomic(tmp_path: Path) -> None:
    man = mark_cell_complete(
        {"version": 1, "completed": {}},
        fold_id=0,
        seed=7,
        arm="opt",
        pnl={"2015-01-02": 0.01},
    )
    save_manifest(tmp_path, man)
    loaded = load_manifest(tmp_path)
    assert is_cell_complete(loaded, 0, 7, "opt")
    assert not is_cell_complete(loaded, 1, 7, "opt")
    assert loaded["completed"][cell_key(0, 7, "opt")]["pnl"]["2015-01-02"] == pytest.approx(0.01, **FLOAT_TOL)


def test_purge_orphan_fold_cells_removes_fold_cache_without_seed_artifact(
    tmp_path: Path,
) -> None:
    manifest = mark_cell_complete(
        {"version": 1, "completed": {}},
        fold_id=3,
        seed=17,
        arm="eq",
        pnl={"2020-01-02": 0.01},
    )

    removed = purge_orphan_fold_cells(
        manifest,
        out_dir=tmp_path,
        arm="eq",
        seeds=[17],
    )

    assert removed == [cell_key(3, 17, "eq")]
    assert not is_cell_complete(manifest, 3, 17, "eq")


def test_run_cpcv_skips_completed_fold_on_resume(tmp_path: Path) -> None:
    dates = list(pd.bdate_range("2015-01-01", periods=60))
    cfg = CPCVConfig(n_splits=3, n_test_groups=1, purge_days=0, embargo_days=0)
    seed, arm = 42, "opt"

    # Pretend fold 0 already finished.
    man = mark_cell_complete(
        {"version": 1, "completed": {}},
        fold_id=0,
        seed=seed,
        arm=arm,
        pnl={"2015-01-02": 0.5},
    )
    save_manifest(tmp_path, man)

    calls: list[int] = []

    def fold_runner(fold: CPCVFold) -> dict[str, float]:
        calls.append(int(fold.fold_id))
        return {str(pd.Timestamp(dates[0]).date()): float(fold.fold_id)}

    res = run_cpcv(
        dates,
        fold_runner,
        cfg,
        resume=True,
        out_dir=tmp_path,
        seed=seed,
        arm=arm,
    )
    assert 0 not in calls, "completed fold 0 must be skipped"
    assert calls, "remaining folds should still run"
    # Cached pnl for fold 0 restored from manifest.
    assert any(f["fold_id"] == 0 for f in res["folds"])
    # Manifest still has fold 0 plus newly completed folds.
    reloaded = load_manifest(tmp_path)
    assert is_cell_complete(reloaded, 0, seed, arm)
    for fid in calls:
        assert is_cell_complete(reloaded, fid, seed, arm)


def test_resume_false_reruns_completed(tmp_path: Path) -> None:
    dates = list(pd.bdate_range("2015-01-01", periods=60))
    cfg = CPCVConfig(n_splits=3, n_test_groups=1, purge_days=0, embargo_days=0)
    man = mark_cell_complete(
        {"version": 1, "completed": {}},
        fold_id=0,
        seed=1,
        arm="eq",
        pnl={"x": 1.0},
    )
    save_manifest(tmp_path, man)
    mock = MagicMock(return_value={"2015-01-02": 0.0})
    run_cpcv(
        dates,
        mock,
        cfg,
        resume=False,
        out_dir=tmp_path,
        seed=1,
        arm="eq",
    )
    fold_ids = [c.args[0].fold_id for c in mock.call_args_list]
    assert 0 in fold_ids


def test_research_alpha_cpcv_forwards_resume_through_run_cpcv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W3.1: out_dir on run_research_alpha_cpcv must reach run_cpcv's fold
    manifest (arm="eq_dii") so a second call with the same out_dir/seed
    retrains zero folds instead of re-paying the whole CPCV budget.
    """
    rng = np.random.default_rng(0)
    t, k = 90, 4
    rets = rng.normal(0.001, 0.01, size=(t, k))
    fac = rng.normal(0.0, 0.005, size=(t, 4))
    dates = pd.bdate_range("2020-01-01", periods=t)
    cfg = {
        "claim_tier": "research",
        "primary_train": "historical_arm_env", "portfolio_arm": "eq",
        "headline_fill": "pct75",
        "n_assets": k,
        "policy": "single_agent",
        "projection_mode": "soft",
        "train_epochs": 1,
        "lr": 3e-4,
    }
    cpcv = CPCVConfig(n_splits=3, n_test_groups=1, purge_days=0, embargo_days=0)

    real_train = research_alpha_cpcv.train_research_hist
    calls = {"n": 0}

    def counting_train(*args, **kwargs):
        calls["n"] += 1
        return real_train(*args, **kwargs)

    monkeypatch.setattr(research_alpha_cpcv, "train_research_hist", counting_train)

    art1 = research_alpha_cpcv.run_research_alpha_cpcv(
        dates, rets, fac, dict(cfg), cpcv=cpcv, seed=0, panel_source="toy",
        out_dir=tmp_path, resume=True,
    )
    assert calls["n"] == cpcv.n_folds(), "first run should train every fold once"
    assert art1["dry_run"] is False

    calls["n"] = 0
    art2 = research_alpha_cpcv.run_research_alpha_cpcv(
        dates, rets, fac, dict(cfg), cpcv=cpcv, seed=0, panel_source="toy",
        out_dir=tmp_path, resume=True,
    )
    assert calls["n"] == 0, "resumed run must not retrain any already-completed fold"
    assert art2["dry_run"] is False
