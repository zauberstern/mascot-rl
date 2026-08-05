"""A1: CPCV folds within a seed must never warm-start from a prior fold.

Warm-starting fold n+1 from fold n's trained agent let the later fold's
parameters be shaped by data overlapping its own test segment (the earlier
fold trained on it), breaking the CPCV "never trained on what it is scored
on" guarantee. Every fold must receive ``agent=None`` so
``train_research_hist`` seeds a brand-new network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mascotrl.eval.research_alpha_cpcv as racpcv
from mascotrl.eval.cpcv import CPCVConfig


def _toy_dates_panel(t: int = 260, k: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = list(pd.bdate_range("2015-01-01", periods=t))
    rets = rng.normal(0.0004, 0.01, size=(t, k))
    fac = rng.normal(0.0, 0.008, size=(t, 4))
    return dates, rets, fac


def _base_cfg() -> dict:
    return {
        "headline_fill": "pct75",
        "primary_train": "historical_arm_env",
        "claim_tier": "research",
        "equity_bps": 5.0,
        "impact_c_eq": 0.0,
        "train_env_steps": 32,
        "train_episodes": 1,
        "train_epochs": 1,
        "n_minibatches": 1,
        "ppo_hidden": 8,
        "rl_backend": "custom",
        "arm": {"id": "eq", "option_slots": 0, "equity_slots": 4, "delta_mode": "off"},
    }


def test_fold_runner_never_passes_warm_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    dates, rets, fac = _toy_dates_panel()
    cfg = _base_cfg()
    cpcv = CPCVConfig(n_splits=3, n_test_groups=1, purge_days=1, embargo_days=1)

    seen_agents: list[object] = []
    real_train = racpcv.train_research_hist

    def _spy(*args, **kwargs):
        seen_agents.append(kwargs.get("agent"))
        return real_train(*args, **kwargs)

    monkeypatch.setattr(racpcv, "train_research_hist", _spy)

    racpcv.run_research_alpha_cpcv(dates, rets, fac, cfg, cpcv=cpcv, seed=0)

    assert len(seen_agents) >= 2, "expected at least two folds with enough train data"
    assert all(a is None for a in seen_agents), (
        f"fold_runner passed a warm-started agent: {seen_agents}"
    )


def test_no_warm_start_folds_key_in_config() -> None:
    """warm_start_folds is no longer a recognized knob; folds are always fresh."""
    import yaml
    from pathlib import Path

    cfg_path = Path("config/workflows/arm_equity.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())
    assert "warm_start_folds" not in cfg


def test_fold_agents_do_not_share_parameter_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    dates, rets, fac = _toy_dates_panel()
    cfg = _base_cfg()
    cpcv = CPCVConfig(n_splits=3, n_test_groups=1, purge_days=1, embargo_days=1)

    trained_agents: list[object] = []
    real_train = racpcv.train_research_hist

    def _spy(*args, **kwargs):
        out = real_train(*args, **kwargs)
        trained_agents.append(out["agent"])
        return out

    monkeypatch.setattr(racpcv, "train_research_hist", _spy)
    racpcv.run_research_alpha_cpcv(dates, rets, fac, cfg, cpcv=cpcv, seed=0)

    assert len(trained_agents) >= 2
    ptrs = [id(a.net) for a in trained_agents]
    assert len(set(ptrs)) == len(ptrs), "fold agents must not share the same network object"
