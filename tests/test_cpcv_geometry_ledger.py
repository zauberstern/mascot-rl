"""CPCV geometry and trial-ledger DSR honesty fences."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.cpcv import CPCVConfig, assign_paths
from src.eval.pbo_appendix import append_trial_ledger_entry
from src.eval.stats_rigor import deflated_sharpe_ratio


def test_cpcv_geometry_6_2_has_15_splits_and_5_paths():
    cfg = CPCVConfig(n_splits=6, n_test_groups=2)
    assert cfg.n_folds() == 15
    assert cfg.n_paths() == 5
    paths = assign_paths(cfg)
    assert len(paths) == 5


def test_trial_ledger_append_changes_dsr_n(tmp_path: Path):
    ledger = tmp_path / "trial_ledger.json"
    append_trial_ledger_entry(
        ledger, source="unit", trial_id="a", sharpe=0.5, status="ok"
    )
    append_trial_ledger_entry(
        ledger, source="unit", trial_id="b", sharpe=0.8, status="ok"
    )
    append_trial_ledger_entry(
        ledger, source="unit", trial_id="c", sharpe=-0.2, status="ok"
    )
    import json

    blob = json.loads(ledger.read_text())
    n = len(blob["trials"])
    assert n == 3
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, size=200)
    dsr1 = deflated_sharpe_ratio(rets, n_trials=1)
    dsr_n = deflated_sharpe_ratio(rets, n_trials=n)
    assert dsr1["dsr"] >= dsr_n["dsr"] - 1e-9
