"""Phase D: training budget fail-closed gates and learning-curve telemetry."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mascotrl.eval.research_alpha_train import train_research_hist
from mascotrl.eval.train_budget import (
    assert_optimizer_step_floor,
    write_learning_curve,
)


def test_assert_optimizer_step_floor():
    assert_optimizer_step_floor(100, min_steps=50)
    with pytest.raises(RuntimeError, match="optimizer_steps"):
        assert_optimizer_step_floor(3, min_steps=10)


def test_train_budget_multi_episode_increases_steps(tmp_path: Path):
    rng = np.random.default_rng(0)
    t, k = 40, 3
    rets = rng.normal(0.0, 0.01, size=(t, k))
    fac = rng.normal(0.0, 0.01, size=(t, 4))
    cfg = {
        "primary_train": "historical_arm_env",
        "policy": "single_agent",
        "headline_fill": "pct75",
        "equity_bps": 5.0,
        "train_epochs": 2,
        "n_minibatches": 2,
        "train_episodes": 3,
        "weight_head": "softmax",
        "arm": {"id": "eq", "option_slots": 0, "equity_slots": k, "delta_mode": "off"},
    }
    out = train_research_hist(rets, fac, cfg, seed=0)
    assert out["n_episodes"] == 3
    assert out["n_steps"] > (t - 2)  # more than one episode
    assert out["optimizer_steps"] >= 2
    assert len(out["learning_curve"]) == 3
    path = write_learning_curve(out["learning_curve"], tmp_path / "curve.json")
    assert path.is_file()
    loaded = json.loads(path.read_text())
    assert len(loaded) == 3


def test_warm_start_reuses_agent_weights():
    rng = np.random.default_rng(1)
    t, k = 30, 3
    rets = rng.normal(0.0, 0.01, size=(t, k))
    fac = rng.normal(0.0, 0.01, size=(t, 4))
    cfg = {
        "primary_train": "historical_arm_env",
        "policy": "single_agent",
        "headline_fill": "pct75",
        "equity_bps": 5.0,
        "train_epochs": 1,
        "n_minibatches": 1,
        "train_episodes": 1,
        "weight_head": "softmax",
        "rl_backend": "custom",
        "arm": {"id": "eq", "option_slots": 0, "equity_slots": k, "delta_mode": "off"},
    }
    out1 = train_research_hist(rets, fac, cfg, seed=0)
    agent = out1["agent"]
    w0 = [p.detach().clone() for p in agent.net.parameters()]
    out2 = train_research_hist(rets, fac, cfg, seed=1, agent=agent)
    # Same object returned.
    assert out2["agent"] is agent
    # Weights should have moved after second train.
    moved = False
    for a, b in zip(w0, agent.net.parameters()):
        if not torch_allclose(a, b.detach()):
            moved = True
            break
    assert moved


def torch_allclose(a, b) -> bool:
    import torch

    return bool(torch.allclose(a, b))
