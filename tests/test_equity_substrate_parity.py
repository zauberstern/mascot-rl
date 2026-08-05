"""TDD: spectrum observes the same equity substrate as H0 (cube + geometry_lite)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_stamp_equity_obs_defaults_enables_cube_and_surface_lane() -> None:
    from mascotrl.eval.equity_substrate import stamp_equity_obs_defaults

    cfg: dict = {"architecture": "mlp", "use_surface_signals": True}
    out = stamp_equity_obs_defaults(cfg)
    assert out["use_equity_feature_cube"] is True
    assert int(out["feature_seq_len"]) == 1
    assert out["surface_obs_lane"] == "geometry_lite"


def test_resolve_spectrum_budget_uses_train_env_steps() -> None:
    from scripts.run_spectrum_campaign import resolve_spectrum_budget

    cfg = {
        "claim_tier": "research",
        "protocol_tier": "screening",
        "algo": "ppo",
        "seeds": [0],
        "train_env_steps": 25_000,
        "cpcv_n_splits": 6,
        "cpcv_n_test_groups": 2,
    }
    bud = resolve_spectrum_budget(cfg)
    # Stamp must not collapse to the old default of 1 when train_env_steps is set.
    assert int(bud["n_episodes"]) >= 1
    assert int(bud["train_env_steps"]) == 25_000
    assert int(bud["n_episodes"]) > 1 or int(bud.get("horizon") or 0) > 0
    # Honest episode estimate from steps (panel length unknown → keep steps stamp).
    assert bud["train_env_steps"] == 25_000


def test_schema_allows_surface_obs_lane_and_obs_pack_path() -> None:
    from mascotrl.spectrum.cell_schema import validate_cell_cfg

    cfg = {
        "spectrum_cell_id": "eq_K4_single_ppo_mlp_softmax_mean_std_cao",
        "portfolio_arm": "eq",
        "n_assets": 4,
        "algo": "ppo",
        "policy_algo": "ppo",
        "architecture": "mlp",
        "temporal_backend": "mlp",
        "weight_head": "softmax",
        "head_axis_id": "softmax",
        "objective": "mean_std_cao",
        "train_world": "historical",
        "train_distribution": "historical",
        "policy_mode": "shared",
        "agent": "single",
        "policy": "single_agent",
        "action_law": "softmax",
        "protocol_tier": "screening",
        "seeds": [0],
        "train_env_steps": 100,
        "cpcv_n_splits": 3,
        "cpcv_n_test_groups": 1,
        "cpcv_purge_days": 0,
        "cpcv_embargo_days": 0,
        "use_equity_feature_cube": True,
        "use_surface_signals": True,
        "surface_obs_lane": "geometry_lite",
        "obs_pack_path": "config/obs_packs/surf_geometry_lite.yaml",
    }
    validate_cell_cfg(cfg, path="test")


def test_attach_obs_substrate_injects_iv_surface_and_dollar_volume() -> None:
    from mascotrl.eval.equity_substrate import attach_equity_obs_substrate
    from mascotrl.features.blocks.obs_builder import PanelObservationBuilder

    T, K = 40, 4
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0, 0.01, size=(T, K))
    dates = pd.bdate_range("2020-01-01", periods=T)
    secids = [100 + i for i in range(K)]
    slots_rows = [list(secids) for _ in range(T)]
    # Synthetic long surface table (geometry_lite channels).
    rows = []
    for d in dates[::5]:
        for s in secids:
            rows.append(
                {
                    "secid": s,
                    "date": d,
                    "mfiv_30": 0.2,
                    "iv_term_slope": 0.01,
                    "iv_skew_30d": 0.03,
                }
            )
    signals_long = pd.DataFrame(rows)
    dollar_volume = np.abs(rng.normal(1e6, 1e5, size=(T, K)))

    cfg: dict = {
        "use_equity_feature_cube": True,
        "use_surface_signals": True,
        "surface_obs_lane": "geometry_lite",
        "_slots_rows": slots_rows,
        "_universe_secids": secids,
    }
    meta = attach_equity_obs_substrate(
        cfg,
        dates=list(dates),
        rets=rets,
        secids=secids,
        slots_rows=slots_rows,
        dollar_volume=dollar_volume,
        signals_long=signals_long,
    )
    extras = cfg["feature_extras"]
    assert "iv_surface" in extras
    assert set(extras["iv_surface"]) >= {"mfiv_30", "iv_term_slope", "iv_skew_30d"}
    assert extras["dollar_volume"].shape == (T, K)
    assert meta["surface_attached"] is True
    assert meta["n_surface_channels"] == 3

    builder = PanelObservationBuilder(rets, extras=extras, seq_len=1, normalize=True)
    obs = builder(10, np.full(K, 1.0 / K))
    # Raw-return fallback would be dim K; cube must be strictly larger.
    assert obs.shape[-1] > K
    # H0-like: static channels include surface(3) + liquidity(>=1) + portfolio(4).
    assert builder.obs_channels_per_asset >= 21


def test_attach_obs_substrate_fail_closed_without_surface_when_requested(
    tmp_path: Path,
) -> None:
    from mascotrl.eval.equity_substrate import attach_equity_obs_substrate

    T, K = 20, 3
    rets = np.zeros((T, K))
    dates = list(pd.bdate_range("2020-01-01", periods=T))
    secids = list(range(K))
    empty_lake = tmp_path / "empty_lake"
    empty_lake.mkdir()
    cfg = {
        "use_equity_feature_cube": True,
        "use_surface_signals": True,
        "surface_obs_lane": "geometry_lite",
        "_slots_rows": [secids] * T,
        "_lake_root": str(empty_lake),
        "lake_root": str(empty_lake),
    }
    with pytest.raises(ValueError, match="surface"):
        attach_equity_obs_substrate(
            cfg,
            dates=dates,
            rets=rets,
            secids=secids,
            slots_rows=[secids] * T,
            dollar_volume=None,
            signals_long=None,
            lake_root=empty_lake,
        )


def test_lake_root_alias_prefers_lake_base_env(monkeypatch) -> None:
    from mascotrl.eval import equity_substrate as es

    monkeypatch.delenv("MASCOTRL_LAKE_DIR", raising=False)
    monkeypatch.setenv("MASCOTRL_LAKE_BASE", "/tmp/fake_lake_base_for_parity")
    root = es.resolve_lake_root()
    assert str(root) == "/tmp/fake_lake_base_for_parity"


def test_spectrum_campaign_exports_stamp_helper() -> None:
    """Campaign must import stamp_equity_obs_defaults (wired into research arm)."""
    from scripts import run_spectrum_campaign as camp

    assert hasattr(camp, "stamp_equity_obs_defaults")
    cfg = {"architecture": "mlp", "portfolio_arm": "eq", "n_assets": 4}
    out = camp.stamp_equity_obs_defaults(cfg)
    assert out["use_equity_feature_cube"] is True


def test_run_happo_cpcv_eq_historical_uses_lake_substrate(monkeypatch) -> None:
    """HAPPO narrative CPCV must not silently keep Arctic OM for eq historical."""
    import numpy as np
    import pandas as pd
    from mascotrl.eval import research_happo_cpcv as hap

    calls: list[str] = []

    def fake_lake(cfg, k):
        calls.append("lake")
        T, K = 80, int(k)
        dates = list(pd.bdate_range("2014-01-01", periods=T))
        rets = np.zeros((T, K), dtype=np.float64)
        factors = np.zeros((T, 4), dtype=np.float64)
        cfg["_universe_secids"] = list(range(K))
        cfg["_slots_rows"] = [list(range(K)) for _ in range(T)]
        cfg["_slot_valid_mask"] = np.ones((T, K), dtype=bool)
        return dates, rets, factors, {"panel_source": "lake_sp500_sec", "k": K}

    def boom_om(*_a, **_k):
        calls.append("om")
        raise AssertionError("OM panel must not be used for eq historical HAPPO")

    monkeypatch.setattr(
        "mascotrl.eval.equity_substrate.load_lake_dyn_hrp_panel", fake_lake
    )
    # Patch the import site used inside run_happo_cpcv.
    import scripts.run_spectrum_campaign as camp

    monkeypatch.setattr(camp, "_try_om_research_panel", boom_om)

    def fake_cpcv_runner(dates, fold_runner, cpcv, **kwargs):
        return {"path_summary": {"n_paths": 1, "sharpe_mean": 0.0}}

    monkeypatch.setattr(hap, "run_cpcv", fake_cpcv_runner)
    monkeypatch.setattr(hap, "resolve_use_purgedcv", lambda _cfg: False)

    cfg = {
        "algo": "happo",
        "portfolio_arm": "eq",
        "train_world": "historical",
        "n_assets": 4,
        "architecture": "mlp",
        "cpcv_n_splits": 2,
        "cpcv_n_test_groups": 1,
        "cpcv_purge_days": 0,
        "cpcv_embargo_days": 0,
        "seeds": [0],
        "use_surface_signals": True,
    }
    budget = {
        "claim_tier": "narrative",
        "cpcv_n_splits": 2,
        "cpcv_n_test_groups": 1,
        "seeds": [0],
        "n_episodes": 2,
        "horizon": 6,
        "dispatch_only": False,
    }
    art, err = hap.run_happo_cpcv(
        cfg, "eq", budget=budget, allow_toy_panel=False, no_dry_run=True
    )
    assert err is None, err
    assert art is not None
    assert art["panel_source"] == "lake_sp500_sec"
    assert calls == ["lake"]


def test_run_happo_arm_eq_historical_prefers_lake(monkeypatch) -> None:
    """Dispatch-only HAPPO eq arm must prefer lake returns over Arctic OM."""
    import numpy as np
    import pandas as pd
    from scripts import run_spectrum_campaign as camp

    calls: list[str] = []

    def fake_lake(cfg, k):
        calls.append("lake")
        T, K = 40, int(k)
        dates = list(pd.bdate_range("2014-01-01", periods=T))
        rets = np.random.default_rng(0).normal(0, 0.01, size=(T, K))
        factors = np.zeros((T, 4))
        return dates, rets, factors, {"panel_source": "lake_sp500_sec", "k": K}

    def boom_om(*_a, **_k):
        calls.append("om")
        raise AssertionError("OM must not win over lake for eq historical")

    monkeypatch.setattr(
        "mascotrl.eval.equity_substrate.load_lake_dyn_hrp_panel", fake_lake
    )
    monkeypatch.setattr(camp, "_try_om_research_panel", boom_om)

    cfg = {
        "algo": "happo",
        "train_world": "historical",
        "n_assets": 4,
        "n_steps": 16,
        "d_model": 16,
        "d_state": 8,
        "macro_dim": 8,
        "architecture": "mlp",
        "temporal_backend": "mlp",
        "use_dhgnn": False,
        "spatial_mode": "none",
        "spectrum_happo_horizon": 4,
        "spectrum_happo_episodes": 1,
        "seed": 0,
        "execution_spread_bps": 5.0,
        "cost_in_decision": True,
    }
    art, err = camp._run_happo_arm(cfg, "eq")
    assert err is None, err
    assert art is not None
    assert art["panel_source"] == "lake_sp500_sec"
    assert calls == ["lake"]
