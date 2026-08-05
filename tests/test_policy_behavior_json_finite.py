"""Strict JSON serialization for policy behavior exports."""
from __future__ import annotations

import pytest
from tests.conftest import FLOAT_TOL

import json
from pathlib import Path

from mascotrl.reporting.policy_behavior import write_policy_behavior


def test_write_policy_behavior_uses_null_not_nan(tmp_path: Path) -> None:
    payload = {
        "behaviour": {"turnover_mean": float("nan"), "holding_period_days": 5.0},
        "archetype_margin": float("nan"),
    }
    path = write_policy_behavior(tmp_path / "behaviour.json", payload)
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    loaded = json.loads(text)
    assert loaded["behaviour"]["turnover_mean"] is None
    assert loaded["behaviour"]["holding_period_days"] == pytest.approx(5.0, **FLOAT_TOL)
    assert loaded["archetype_margin"] is None
