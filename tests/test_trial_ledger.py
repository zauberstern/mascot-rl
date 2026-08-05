"""Append-only trial ledger (Alpha v2 Block D Step 22)."""
from __future__ import annotations

from pathlib import Path

import pytest

from mascotrl.eval.trial_ledger import TrialLedger, append_trial, load_ledger


def test_append_baseline_seed_fold_even_on_failure(tmp_path: Path):
    path = tmp_path / "trials.jsonl"
    ledger = TrialLedger(path)
    ledger.append(
        baseline="ridge",
        seed=100,
        fold=0,
        status="ok",
        sharpe=0.4,
    )
    ledger.append(
        baseline="no_trade",
        seed=101,
        fold=1,
        status="failed",
        error="oom",
    )
    rows = ledger.rows()
    assert len(rows) == 2
    assert rows[0]["baseline"] == "ridge"
    assert rows[0]["seed"] == 100
    assert rows[0]["fold"] == 0
    assert rows[0]["status"] == "ok"
    assert rows[1]["status"] == "failed"
    assert rows[1]["error"] == "oom"

    # Reload preserves both success and failure rows.
    reloaded = load_ledger(path)
    assert len(reloaded) == 2
    assert reloaded[1]["status"] == "failed"


def test_refuse_mutation_of_past_rows(tmp_path: Path):
    path = tmp_path / "trials.json"
    ledger = TrialLedger(path)
    ledger.append(baseline="equal_weight", seed=0, fold=0, status="ok")
    rows = ledger.rows()
    with pytest.raises((TypeError, AttributeError, ValueError)):
        rows[0]["sharpe"] = 99.0
    # Even if caller mutates a copy, disk / ledger state must stay intact.
    snap = list(ledger.rows())
    try:
        snap[0] = {**snap[0], "sharpe": 99.0}
    except Exception:
        pass
    again = ledger.rows()
    assert "sharpe" not in again[0] or again[0].get("sharpe") != 99.0

    # No overwrite / replace API for historical indices.
    with pytest.raises((TypeError, AttributeError, ValueError)):
        ledger.replace(0, baseline="hacked", seed=0, fold=0, status="ok")


def test_append_trial_helper(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    append_trial(
        path,
        baseline="ridge",
        seed=7,
        fold=2,
        status="ok",
        metrics={"sharpe": 0.12},
    )
    rows = load_ledger(path)
    assert len(rows) == 1
    assert rows[0]["baseline"] == "ridge"
    assert rows[0]["seed"] == 7
    assert rows[0]["fold"] == 2
