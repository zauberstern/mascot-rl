"""TDD: A-1 secid fail-closed, A-2 surface NaN gate, A-3 cube attach swallow."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_resolve_substrate_secids_toy_allows_range_fallback() -> None:
    from mascotrl.eval.equity_substrate import resolve_substrate_secids

    out = resolve_substrate_secids({}, panel_source="toy", k=4)
    assert out == [0, 1, 2, 3]


def test_resolve_substrate_secids_lake_raises_without_stamp() -> None:
    from mascotrl.eval.equity_substrate import resolve_substrate_secids

    with pytest.raises(RuntimeError, match="_universe_secids"):
        resolve_substrate_secids({}, panel_source="lake_sp500_sec", k=4)


def test_resolve_substrate_secids_optionmetrics_raises_without_stamp() -> None:
    from mascotrl.eval.equity_substrate import resolve_substrate_secids

    with pytest.raises(RuntimeError, match="_universe_secids"):
        resolve_substrate_secids({}, panel_source="optionmetrics", k=4)


def test_stamp_lake_universe_secids_for_featnet_noop_when_present() -> None:
    from mascotrl.eval.equity_substrate import stamp_lake_universe_secids_for_featnet

    cfg = {"_universe_secids": ["A", "B"]}
    out = stamp_lake_universe_secids_for_featnet(cfg, k=2)
    assert out == ["A", "B"]


def test_stamp_lake_universe_secids_for_featnet_loads_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mascotrl.eval import equity_substrate as es

    def _fake_load(cfg, *, k: int, lake_root=None, max_pool: int = 400):
        cfg["_universe_secids"] = [f"S{i}" for i in range(int(k))]
        cfg["_slots_rows"] = [["S0"] * int(k)]
        cfg["_lake_root"] = "/tmp/fake_lake"
        return [], None, None, {}

    monkeypatch.setattr(es, "load_lake_dyn_hrp_panel", _fake_load)
    cfg: dict = {}
    out = es.stamp_lake_universe_secids_for_featnet(cfg, k=3)
    assert out == ["S0", "S1", "S2"]
    assert cfg["_universe_secids"] == ["S0", "S1", "S2"]
    # Physics path had no slots: lake slots must not leak onto OM dates.
    assert "_slots_rows" not in cfg


def test_resolve_substrate_secids_lake_uses_stamp() -> None:
    from mascotrl.eval.equity_substrate import resolve_substrate_secids

    secids = [101, 102, 103]
    out = resolve_substrate_secids(
        {"_universe_secids": secids}, panel_source="lake_sp500_sec", k=3
    )
    assert out == secids


def test_attach_surface_nan_fail_closed_above_threshold() -> None:
    from mascotrl.eval.equity_substrate import attach_equity_obs_substrate

    T, K = 100, 4
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0, 0.01, size=(T, K))
    dates = pd.bdate_range("2020-01-01", periods=T)
    secids = [100 + i for i in range(K)]
    slots_rows = [list(secids) for _ in range(T)]

    rows = []
    for i, d in enumerate(dates):
        for s in secids:
            rows.append(
                {
                    "secid": s,
                    "date": d,
                    "mfiv_30": np.nan if i < 25 else 0.2,
                    "iv_term_slope": 0.01,
                    "iv_skew_30d": 0.03,
                }
            )
    signals_long = pd.DataFrame(rows)

    cfg: dict = {
        "use_equity_feature_cube": True,
        "use_surface_signals": True,
        "surface_obs_lane": "geometry_lite",
        "_slots_rows": slots_rows,
        "_universe_secids": secids,
    }
    with pytest.raises(SystemExit, match="mfiv_30"):
        attach_equity_obs_substrate(
            cfg,
            dates=list(dates),
            rets=rets,
            secids=secids,
            slots_rows=slots_rows,
            signals_long=signals_long,
        )


def test_run_research_arm_cube_attach_failure_raises_on_lake(monkeypatch) -> None:
    from scripts import run_spectrum_campaign as camp

    def fake_lake(cfg, k):
        T, K = 40, int(k)
        dates = list(pd.bdate_range("2014-01-01", periods=T))
        rets = np.zeros((T, K), dtype=np.float64)
        factors = np.zeros((T, 4), dtype=np.float64)
        secids = list(range(K))
        cfg["_universe_secids"] = secids
        cfg["_slots_rows"] = [secids for _ in range(T)]
        return dates, rets, factors, {"panel_source": "lake_sp500_sec", "k": K}

    def boom_attach(*_a, **_k):
        raise RuntimeError("substrate attach boom")

    monkeypatch.setattr(
        "mascotrl.eval.equity_substrate.load_lake_dyn_hrp_panel", fake_lake
    )
    monkeypatch.setattr(
        "mascotrl.eval.equity_substrate.attach_equity_obs_substrate", boom_attach
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
        "use_equity_feature_cube": True,
        "use_surface_signals": False,
    }
    with pytest.raises(RuntimeError, match="substrate attach boom"):
        camp._run_research_arm(cfg, "eq", allow_toy_panel=False, no_dry_run=True)


def test_run_research_arm_cube_attach_failure_soft_on_toy(monkeypatch) -> None:
    from scripts import run_spectrum_campaign as camp

    monkeypatch.setattr(camp, "_try_om_research_panel", lambda cfg, k: None)

    def boom_lake(*_a, **_k):
        raise FileNotFoundError("lake missing")

    def boom_attach(*_a, **_k):
        raise RuntimeError("substrate attach boom")

    monkeypatch.setattr(
        "mascotrl.eval.equity_substrate.load_lake_dyn_hrp_panel", boom_lake
    )
    monkeypatch.setattr(
        "mascotrl.eval.equity_substrate.attach_equity_obs_substrate", boom_attach
    )

    def fake_cpcv(*_a, **_k):
        return {"path_summary": {"sharpe_mean": 0.1}, "panel_source": "toy"}

    monkeypatch.setattr(
        "mascotrl.eval.research_alpha_cpcv.run_research_alpha_cpcv", fake_cpcv
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
        "use_equity_feature_cube": True,
        "use_surface_signals": False,
    }
    art, err = camp._run_research_arm(
        cfg, "eq", allow_toy_panel=True, no_dry_run=True
    )
    assert err is None
    assert art is not None
    assert art.get("_feature_net_errors") == ["substrate attach boom"]
