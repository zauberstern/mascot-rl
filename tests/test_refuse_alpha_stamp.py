"""Alpha v2 Block E Step 28: refuse alpha stamp without G0 / SPA / ledger."""
from __future__ import annotations

import copy

import pytest

from src.eval.gate_ladder import refuse_alpha_stamp


def _ok_report() -> dict:
    return {
        "arm": "eq",
        "alpha_claim": True,
        "gate_ladder": {
            "pass": True,
            "gates": {
                "G0": {"pass": True},
                "G1": {"pass": True},
                "G2": {"pass": True},
                "G3": {"pass": True},
                "G4": {"pass": True},
                "G5": {"pass": True},
                "G6": {"pass": True},
            },
        },
        "hansen_spa": {
            "ok": True,
            "n_economic_rivals": 4,
            "rival_names": [
                "no_trade",
                "ridge",
                "rv_iv_rank",
                "best_single_agent_rl",
            ],
        },
        "trial_ledger": {
            "complete": True,
            "trials": [
                {"baseline": "ridge", "seed": 0, "fold": 0, "status": "ok"},
                {"baseline": "no_trade", "seed": 1, "fold": 0, "status": "ok"},
            ],
        },
    }


def test_refuse_alpha_stamp_passes_when_clean():
    refuse_alpha_stamp(_ok_report())  # does not raise


def test_refuse_when_g0_false():
    report = copy.deepcopy(_ok_report())
    report["gate_ladder"]["gates"]["G0"]["pass"] = False
    report["gate_ladder"]["pass"] = False
    with pytest.raises(ValueError, match="G0"):
        refuse_alpha_stamp(report)


def test_refuse_when_spa_rivals_thin():
    report = copy.deepcopy(_ok_report())
    report["hansen_spa"] = {
        "ok": False,
        "reason": "spa_rivals_insufficient",
        "n_economic_rivals": 1,
    }
    with pytest.raises(ValueError, match="SPA|rivals"):
        refuse_alpha_stamp(report)


def test_refuse_when_ledger_incomplete():
    report = copy.deepcopy(_ok_report())
    report["trial_ledger"] = {"complete": False, "trials": []}
    with pytest.raises(ValueError, match="ledger"):
        refuse_alpha_stamp(report)
