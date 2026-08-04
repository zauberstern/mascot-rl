"""CPCV resume must not mark failed folds complete."""
from __future__ import annotations

from pathlib import Path

from src.eval.campaign_manifest import (
    is_cell_complete,
    load_manifest,
    mark_cell_complete,
    save_manifest,
)


def test_empty_pnl_is_not_complete(tmp_path: Path):
    man = load_manifest(tmp_path)
    mark_cell_complete(man, 0, 0, "eq", pnl={})
    save_manifest(tmp_path, man)
    man2 = load_manifest(tmp_path)
    assert is_cell_complete(man2, 0, 0, "eq") is False


def test_nonempty_pnl_is_complete(tmp_path: Path):
    man = load_manifest(tmp_path)
    mark_cell_complete(man, 1, 0, "eq", pnl={"2020-01-02": 0.01})
    save_manifest(tmp_path, man)
    assert is_cell_complete(load_manifest(tmp_path), 1, 0, "eq") is True


def test_resume_restores_oos_weight_aux(tmp_path: Path) -> None:
    """D5: kill-and-resume must keep path-0 weight rows for behaviour_export."""
    import numpy as np
    import pandas as pd

    from src.eval.cpcv import CPCVConfig, _CPCV_FOLD_AUX_KEY, group_bounds, run_cpcv

    dates = list(pd.bdate_range("2020-01-01", periods=30))
    bounds = group_bounds(dates, 3)
    calls = {"n": 0}

    def fold_runner(fold):
        calls["n"] += 1
        pnl = {}
        aux = {}
        for g in fold.test_groups:
            lo, hi = bounds[g]
            for i in range(lo, hi + 1):
                ds = str(dates[i].date())
                pnl[ds] = 0.001 * (i + 1)
                aux[ds] = {
                    "total_net": pnl[ds],
                    "weights": [0.5, 0.5],
                    "turnover": 0.1,
                    "cost": 0.0,
                    "gross": pnl[ds],
                }
        out = dict(pnl)
        out[_CPCV_FOLD_AUX_KEY] = aux
        return out

    cfg = CPCVConfig(n_splits=3, n_test_groups=1, purge_days=0, embargo_days=0)
    art1 = run_cpcv(
        dates, fold_runner, cfg, resume=True, out_dir=tmp_path, seed=0, arm="eq"
    )
    assert calls["n"] == cfg.n_folds()
    assert art1["fold_aux"], "first run must cache OOS aux per fold"
    for aux in art1["fold_aux"].values():
        assert any(rec.get("weights") for rec in aux.values())

    calls["n"] = 0
    art2 = run_cpcv(
        dates, fold_runner, cfg, resume=True, out_dir=tmp_path, seed=0, arm="eq"
    )
    assert calls["n"] == 0, "resume must skip all completed folds"
    assert art2["fold_aux"], "resume must restore OOS aux from manifest"
    assert len(art2["fold_aux"]) == len(art1["fold_aux"])
    # Spot-check one weight row survives JSON round-trip.
    fid = next(iter(art2["fold_aux"]))
    sample = next(iter(art2["fold_aux"][fid].values()))
    assert sample["weights"] == [0.5, 0.5]
    assert np.isfinite(float(sample["total_net"]))
