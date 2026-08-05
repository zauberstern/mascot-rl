"""Phase 1-2: HAPPO spectrum budget honesty and CPCV dispatch."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import torch

from scripts.run_spectrum_campaign import (
    _lagged_w_prev_actions,
    _run_happo_arm,
    resolve_spectrum_budget,
)


def test_screening_happo_is_dispatch_only() -> None:
    b = resolve_spectrum_budget(
        {
            "algo": "happo",
            "claim_tier": "research",
            "protocol_tier": "screening",
            "train_env_steps": 25000,
        }
    )
    assert b["dispatch_only"] is True
    assert b["claim_tier"] == "dispatch_only"


def test_narrative_happo_is_not_dispatch_only() -> None:
    b = resolve_spectrum_budget(
        {
            "algo": "happo",
            "claim_tier": "narrative",
            "protocol_tier": "narrative",
            "train_env_steps": 100000,
        }
    )
    assert b["dispatch_only"] is False
    assert b["claim_tier"] in ("narrative", "research")


def test_happo_trainbatch_w_prev_is_lagged() -> None:
    actions = [
        torch.tensor([[0.1, 0.2, 0.3]]),
        torch.tensor([[0.4, 0.5, 0.6]]),
        torch.tensor([[0.7, 0.8, 0.9]]),
    ]
    w_prev = _lagged_w_prev_actions(actions)
    assert w_prev.shape == (3, 3)
    assert torch.allclose(w_prev[0], torch.zeros(3))
    assert torch.allclose(w_prev[1], actions[0])
    assert torch.allclose(w_prev[2], actions[1])


def test_happo_arm_refuses_cost_in_decision_with_zero_spread(monkeypatch) -> None:
    import mascotrl.eval.equity_substrate as es

    monkeypatch.setattr(
        es,
        "load_lake_dyn_hrp_panel",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("lake_disabled_in_unit_test")),
    )
    cfg = {
        "algo": "happo",
        "train_world": "historical",
        "n_assets": 4,
        "d_model": 16,
        "d_state": 8,
        "macro_dim": 8,
        "architecture": "mlp",
        "spectrum_happo_horizon": 4,
        "spectrum_happo_episodes": 1,
        "claim_tier": "dispatch_only",
        "cost_in_decision": True,
        "execution_spread_bps": 0.0,
        "execution_impact_coef": 0.0,
        "seed": 0,
    }
    art, err = _run_happo_arm(cfg, "eq")
    assert art is None
    assert err == "cost_in_decision_requires_nonzero_friction"


def test_screening_happo_smoke_stamps_no_real_reference(monkeypatch) -> None:
    import mascotrl.eval.equity_substrate as es

    monkeypatch.setattr(
        es,
        "load_lake_dyn_hrp_panel",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("lake_disabled_in_unit_test")),
    )
    cfg = {
        "algo": "happo",
        "claim_tier": "research",
        "protocol_tier": "screening",
        "train_world": "historical",
        "n_assets": 4,
        "d_model": 16,
        "d_state": 8,
        "macro_dim": 8,
        "architecture": "mlp",
        "spectrum_happo_horizon": 6,
        "spectrum_happo_episodes": 1,
        "seed": 0,
    }
    budget = resolve_spectrum_budget(cfg)
    cfg["claim_tier"] = budget["claim_tier"]
    cfg["spectrum_happo_episodes"] = budget["n_episodes"]
    cfg["spectrum_happo_horizon"] = budget["horizon"]
    art, err = _run_happo_arm(cfg, "eq")
    assert err is None, err
    assert art is not None
    assert art["real_reference_arm_present"] is False
    assert art["claim_tier"] == "dispatch_only"


def test_narrative_happo_runs_cpcv_not_smoke(monkeypatch) -> None:
    from mascotrl.eval import research_happo_cpcv

    called = {"cpcv": False, "smoke": False}

    def fake_cpcv(cfg, arm, *, budget, allow_toy_panel=False, no_dry_run=False, **_kw):
        called["cpcv"] = True
        return (
            {
                "cpcv": {"n_splits": 3, "config": {"n_splits": 3}},
                "path_summary": {"sharpe_mean": 0.1, "n_paths": 1},
                "real_reference_arm_present": True,
            },
            None,
        )

    def fake_smoke(cfg, arm):
        called["smoke"] = True
        return ({"horizon": 6, "real_reference_arm_present": False}, None)

    monkeypatch.setattr(research_happo_cpcv, "run_happo_cpcv", fake_cpcv)
    with patch("scripts.run_spectrum_campaign._run_happo_arm", fake_smoke):
        from scripts.run_spectrum_campaign import _run_research_arm

        cfg = {
            "algo": "happo",
            "protocol_tier": "narrative",
            "claim_tier": "research",
            "train_world": "historical",
            "n_assets": 4,
            "headline_fill": "pct75",
            "cpcv_n_splits": 3,
            "cpcv_n_test_groups": 1,
        }
        art, err = _run_research_arm(cfg, "eq", allow_toy_panel=True)
    assert err is None, err
    assert called["cpcv"] is True
    assert called["smoke"] is False
    assert art is not None
    assert art["real_reference_arm_present"] is True


def test_run_happo_cpcv_toy_geometry(monkeypatch) -> None:
    import mascotrl.eval.equity_substrate as es
    from mascotrl.eval.research_happo_cpcv import run_happo_cpcv

    # Force the toy-panel branch (lake parity covered elsewhere).
    monkeypatch.setattr(
        es,
        "load_lake_dyn_hrp_panel",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("lake_disabled_in_unit_test")),
    )
    cfg = {
        "algo": "happo",
        "train_world": "historical",
        "n_assets": 4,
        "d_model": 16,
        "d_state": 8,
        "macro_dim": 8,
        "architecture": "mlp",
        "temporal_backend": "mlp",
        "use_dhgnn": False,
        "spatial_mode": "none",
        "cpcv_n_splits": 2,
        "cpcv_n_test_groups": 1,
        "cpcv_purge_days": 0,
        "cpcv_embargo_days": 0,
        "train_env_steps": 8,
        "seeds": [0],
        "seed": 0,
    }
    budget = resolve_spectrum_budget(cfg)
    art, err = run_happo_cpcv(cfg, "eq", budget=budget, allow_toy_panel=True)
    assert err is None, err
    assert art is not None
    assert art["cpcv"]["config"]["n_splits"] == 2
    assert "path_summary" in art
    assert art.get("dispatch_only") is not True
    assert "collapse_guard" in art
