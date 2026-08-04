"""Seeded golden train regression."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.regression._toy_train import run_toy_train

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "golden" / "seed42_toy_ppo_metrics.json"


@pytest.mark.slow
@pytest.mark.regression
def test_seeded_train_golden(torch_deterministic):
    """Fixed-seed short PPO train lands within tolerance of committed golden."""
    _, metrics = run_toy_train(seed=42, steps=200, epochs=2, return_stats=True)
    payload = {
        "final_mean_reward": metrics["final_mean_reward"],
        "final_entropy": metrics["final_entropy"],
        "weight_l1_vs_ew": metrics["weight_l1_vs_ew"],
        "max_weight": metrics["max_weight"],
        "total_optimizer_steps": metrics["total_optimizer_steps"],
    }
    if os.environ.get("MASCOTRL_UPDATE_GOLDEN") == "1":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        pytest.skip("golden updated; re-run without MASCOTRL_UPDATE_GOLDEN=1")

    assert GOLDEN.is_file(), f"missing golden {GOLDEN}; set MASCOTRL_UPDATE_GOLDEN=1 once"
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    mismatches = []
    for key, exp in expected.items():
        got = payload[key]
        if isinstance(exp, float):
            if not (
                abs(got - exp) <= 0.02 + 0.05 * abs(exp)
            ):
                mismatches.append((key, exp, got))
        else:
            if got != exp:
                mismatches.append((key, exp, got))
    assert not mismatches, f"golden mismatch: {mismatches}\nexpected={expected}\ngot={payload}"
