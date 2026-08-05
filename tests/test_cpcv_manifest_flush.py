"""Regression: batched manifest flush preserves resume correctness."""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from mascotrl.eval.campaign_manifest import load_manifest, manifest_path
from mascotrl.eval.cpcv import CPCVConfig, _CPCV_FOLD_AUX_KEY, run_cpcv


def test_manifest_flush_batches_disk_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MASCOTRL_MANIFEST_FLUSH_EVERY", "2")
    dates = list(pd.bdate_range("2020-01-01", periods=120))
    cfg = CPCVConfig(n_splits=6, n_test_groups=2, purge_days=5, embargo_days=5)
    n_folds = cfg.n_folds()
    calls = {"n": 0}

    def fold_runner(fold):
        calls["n"] += 1
        return {str(pd.Timestamp(dates[0]).date()): 0.01}

    save_calls: list[Path] = []
    real_save = __import__(
        "mascotrl.eval.campaign_manifest", fromlist=["save_manifest"]
    ).save_manifest

    def counting_save(out_dir, manifest):
        save_calls.append(Path(out_dir))
        return real_save(out_dir, manifest)

    with mock.patch(
        "mascotrl.eval.campaign_manifest.save_manifest", side_effect=counting_save
    ):
        art = run_cpcv(
            dates,
            fold_runner,
            cfg,
            resume=True,
            out_dir=tmp_path,
            seed=3,
            arm="eq",
        )

    assert art["n_folds"] == n_folds
    assert calls["n"] == n_folds
    assert len(save_calls) < n_folds
    assert manifest_path(tmp_path).is_file()
    manifest = load_manifest(tmp_path)
    assert len(manifest.get("completed") or {}) == n_folds

    # Second run skips all folds (resume intact after batched flush).
    calls["n"] = 0
    with mock.patch(
        "mascotrl.eval.campaign_manifest.save_manifest", side_effect=counting_save
    ):
        art2 = run_cpcv(
            dates,
            fold_runner,
            cfg,
            resume=True,
            out_dir=tmp_path,
            seed=3,
            arm="eq",
        )
    assert calls["n"] == 0
    assert art2["resume"]["n_skipped"] == n_folds


def test_manifest_flush_preserves_oos_aux(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MASCOTRL_MANIFEST_FLUSH_EVERY", "3")
    pytest.importorskip("purgedcv")
    from mascotrl.eval.cpcv_lib import run_cpcv_lib

    dates = list(pd.bdate_range("2020-01-01", periods=180))
    cfg = CPCVConfig(n_splits=6, n_test_groups=2, purge_days=5, embargo_days=5)

    def fold_runner(fold):
        pnl = {str(pd.Timestamp(dates[0]).date()): 0.01}
        pnl[_CPCV_FOLD_AUX_KEY] = {"fold": fold.fold_id}
        return pnl

    art1 = run_cpcv_lib(
        dates, fold_runner, cfg, resume=True, out_dir=tmp_path, seed=1, arm="eq"
    )
    art2 = run_cpcv_lib(
        dates, fold_runner, cfg, resume=True, out_dir=tmp_path, seed=1, arm="eq"
    )
    assert art2["resume"]["n_skipped"] == art1["n_folds"]
    assert set(art2["fold_aux"]) == set(art1["fold_aux"])
