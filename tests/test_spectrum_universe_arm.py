"""universe_arm dyn_crucible foil produces a non-trivial slot mask."""
from __future__ import annotations

import numpy as np

from scripts.run_spectrum_campaign import _apply_spectrum_universe_arm


def test_dyn_crucible_foil_differs_from_full_mask():
    T, K = 200, 10
    rets = np.zeros((T, K))
    cfg: dict = {"universe_arm": "dyn_crucible"}
    _apply_spectrum_universe_arm(cfg, dates=list(range(T)), rets=rets)
    mask = cfg["_slot_valid_mask"]
    assert mask.shape == (T, K)
    assert mask.dtype == bool
    assert not bool(mask.all())
    assert bool(mask.any())
    assert cfg["_universe_arm_applied"] == "dyn_crucible_spectrum_foil"


def test_dyn_hrp_leaves_mask_unset():
    cfg: dict = {"universe_arm": "dyn_hrp"}
    _apply_spectrum_universe_arm(cfg, dates=[], rets=np.zeros((5, 3)))
    assert "_slot_valid_mask" not in cfg


def test_crucible_foil_applied_on_lake_panel():
    """Crucible mask must apply even when panel_source is lake_sp500_sec.

    Regression for RC1: ``_apply_spectrum_universe_arm`` was gated behind
    ``if panel_source != 'lake_sp500_sec'``, so the Sweep I foil was dead
    for all eq historical cells.
    """
    T, K = 200, 10
    rets = np.zeros((T, K))
    # Simulate post-load_lake_dyn_hrp_panel state (HRP mask already present).
    cfg: dict = {
        "universe_arm": "dyn_crucible",
        "_slot_valid_mask": np.ones((T, K), dtype=bool),
        "_universe_arm_applied": "dyn_hrp",
    }
    _apply_spectrum_universe_arm(cfg, dates=list(range(T)), rets=rets)
    mask = cfg["_slot_valid_mask"]
    assert not bool(mask.all()), "crucible mask must drop slots"
    assert bool(mask.any())
    assert cfg["_universe_arm_applied"] == "dyn_crucible_spectrum_foil"
    # Deterministic rotating dropout: (k + q) % 5 == 0 with q = t // 63.
    assert mask[0, 0] is np.False_ or mask[0, 0] == False
    assert mask[0, 1] is np.True_ or mask[0, 1] == True


def test_run_research_arm_applies_crucible_foil_for_eq_lake(monkeypatch):
    """End-to-end: dyn_crucible eq/historical must stamp the spectrum foil."""
    import scripts.run_spectrum_campaign as camp

    T, K = 100, 10
    dates = list(range(T))
    rets = np.zeros((T, K))
    factors = np.zeros((T, 3))
    called: list[str] = []

    def _fake_lake(cfg, k=8):
        cfg.setdefault("universe_arm", "dyn_hrp")
        cfg["_universe_arm_applied"] = str(cfg.get("universe_arm") or "dyn_hrp")
        cfg["_slot_valid_mask"] = np.ones((T, K), dtype=bool)
        cfg["_universe_secids"] = [f"S{i}" for i in range(K)]
        cfg["_slots_rows"] = [[f"S{i}" for i in range(K)] for _ in range(T)]
        return dates, rets, factors, {"panel_source": "lake_sp500_sec", "k": K}

    original_apply = camp._apply_spectrum_universe_arm

    def _spy_apply(cfg, *, dates, rets):
        called.append(str(cfg.get("universe_arm")))
        return original_apply(cfg, dates=dates, rets=rets)

    monkeypatch.setattr(
        "src.eval.equity_substrate.load_lake_dyn_hrp_panel", _fake_lake
    )
    monkeypatch.setattr(
        "src.eval.equity_substrate.stamp_equity_obs_defaults", lambda cfg: None
    )
    monkeypatch.setattr(
        "src.eval.equity_substrate.attach_equity_obs_substrate",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(camp, "_apply_spectrum_universe_arm", _spy_apply)

    def _fake_cpcv(dates, rets, factors, cfg, **_kw):
        return {
            "paths": {"0": {"weights": [[0.1] * K], "turnover": [0.01]}},
            "path_summary": {"sharpe_mean": 0.0},
            "panel_source": "lake_sp500_sec",
            "policy_diagnostics": {},
            "_universe_arm_applied": cfg.get("_universe_arm_applied"),
            "_slot_valid_mask_any_false": (
                not bool(np.asarray(cfg.get("_slot_valid_mask")).all())
                if cfg.get("_slot_valid_mask") is not None
                else None
            ),
        }

    monkeypatch.setattr(
        "src.eval.research_alpha_cpcv.run_research_alpha_cpcv", _fake_cpcv
    )

    cfg = {
        "algo": "ppo",
        "architecture": "mlp",
        "objective": "mean_std_cao",
        "train_world": "historical",
        "portfolio_arm": "eq",
        "universe_arm": "dyn_crucible",
        "n_assets": K,
        "claim_tier": "research",
        "weight_head": "softmax",
        "use_equity_feature_cube": False,
    }
    art, err = camp._run_research_arm(cfg, "eq", allow_toy_panel=True, no_dry_run=False)
    assert err is None, err
    assert "dyn_crucible" in called
    assert art is not None
    assert art.get("_universe_arm_applied") == "dyn_crucible_spectrum_foil"
    assert art.get("_slot_valid_mask_any_false") is True
