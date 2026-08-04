"""DSR trial count must come from the executed campaign ledger."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.run_eq_alloc_campaign import _estimate_campaign_dsr_trials


def test_campaign_dsr_uses_auditable_trial_ledger(tmp_path: Path) -> None:
    ledger = {
        "schema": "mascotrl.trial_ledger.v1",
        "trials": [
            {"source": "eq_alloc_cpcv", "id": f"trial-{i}", "status": "ok"}
            for i in range(7)
        ],
    }
    path = tmp_path / "trial_ledger.json"
    path.write_text(json.dumps(ledger), encoding="utf-8")

    n_trials, meta = _estimate_campaign_dsr_trials(path, cfg={})

    assert n_trials == 7
    assert meta["source"] == "executed_trial_ledger"
    assert meta["auditable"] is True
