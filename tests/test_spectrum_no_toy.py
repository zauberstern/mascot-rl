"""Part D.5: --no-dry-run refuses silent toy panel without --allow-toy-panel."""
from __future__ import annotations

import pytest


def test_no_dry_run_refuses_toy_without_flag(monkeypatch) -> None:
    from scripts import run_spectrum_campaign as rsc

    monkeypatch.setattr(rsc, "_try_om_research_panel", lambda cfg, k: None)

    def _boom(*_a, **_k):
        raise FileNotFoundError("lake missing in unit test")

    monkeypatch.setattr(
        "src.eval.equity_substrate.load_lake_dyn_hrp_panel", _boom
    )
    cfg = {
        "algo": "ppo",
        "architecture": "mlp",
        "objective": "differential_sharpe",
        "train_world": "historical",
        "n_assets": 4,
        "claim_tier": "research",
        "cpcv_n_splits": 3,
        "cpcv_n_test_groups": 1,
    }
    with pytest.raises(RuntimeError, match="refusing toy panel|lake dyn_hrp panel unavailable"):
        rsc._run_research_arm(cfg, "eq", allow_toy_panel=False, no_dry_run=True)


def test_allow_toy_panel_stamps_artifact(monkeypatch) -> None:
    from scripts import run_spectrum_campaign as rsc

    monkeypatch.setattr(rsc, "_try_om_research_panel", lambda cfg, k: None)

    def _boom(*_a, **_k):
        raise FileNotFoundError("lake missing in unit test")

    monkeypatch.setattr(
        "src.eval.equity_substrate.load_lake_dyn_hrp_panel", _boom
    )

    def fake_cpcv(*_a, **_k):
        return {
            "path_summary": {"sharpe_mean": 0.1},
            "policy": {"sharpe_mean": 0.1},
            "panel_source": "toy",
        }

    monkeypatch.setattr(
        "src.eval.research_alpha_cpcv.run_research_alpha_cpcv", fake_cpcv
    )
    cfg = {
        "algo": "ppo",
        "architecture": "mlp",
        "objective": "differential_sharpe",
        "train_world": "historical",
        "n_assets": 4,
        "claim_tier": "research",
        "cpcv_n_splits": 3,
        "cpcv_n_test_groups": 1,
        "seed": 0,
        "use_surface_signals": False,
    }
    art, err = rsc._run_research_arm(
        cfg, "eq", allow_toy_panel=True, no_dry_run=True
    )
    assert err is None
    assert art is not None
    assert art.get("toy_panel") is True
