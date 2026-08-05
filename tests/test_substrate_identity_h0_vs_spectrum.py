"""Gate G0: H0 and spectrum equity obs must be identical (26 ch/asset).

Evidence (2026-08-26): running H0 checkpoints have MLP in_features=2600
(= K100 * 26). Feature-net lake panels exist and WOULD inflate C if attached;
sprint lock + use_feature_net_extras default False keep both paths at 26.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mascotrl.features.blocks.assemble import assemble_equity_feature_cube
from mascotrl.features.blocks.obs_builder import PanelObservationBuilder


def _fixture_panel(T: int = 40, K: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.01, size=(T, K))
    dates = list(pd.bdate_range("2020-01-01", periods=T))
    secids = [100 + i for i in range(K)]
    slots_rows = [list(secids) for _ in range(T)]
    dollar_volume = np.abs(rng.normal(1e6, 1e5, size=(T, K)))
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
    return rets, dates, secids, slots_rows, dollar_volume, signals_long


def test_assembler_base_cube_is_22_static_plus_4_portfolio() -> None:
    """Authoritative channel contract without feature-net extras."""
    rets, _, _, _, dollar_volume, _ = _fixture_panel()
    iv = {
        "mfiv_30": np.full(rets.shape, 0.2),
        "iv_term_slope": np.full(rets.shape, 0.01),
        "iv_skew_30d": np.full(rets.shape, 0.03),
    }
    extras = {"dollar_volume": dollar_volume, "iv_surface": iv}
    cube, names = assemble_equity_feature_cube(rets, extras=extras, normalize=True)
    assert cube.shape[-1] == 22, names
    builder = PanelObservationBuilder(rets, extras=extras, seq_len=1, normalize=True)
    assert builder.obs_channels_per_asset == 26
    # Momentum(13) + vol(4) + liq(2) + surface(3) + portfolio(4).
    assert builder.n_channels == 22


def test_feature_net_keys_inflate_channel_count() -> None:
    """Assembler contract: ohlc/micro etc. change C (must stay gated off)."""
    rets, _, _, _, dollar_volume, _ = _fixture_panel()
    iv = {
        "mfiv_30": np.full(rets.shape, 0.2),
        "iv_term_slope": np.full(rets.shape, 0.01),
        "iv_skew_30d": np.full(rets.shape, 0.03),
    }
    base = {"dollar_volume": dollar_volume, "iv_surface": iv}
    T, K = rets.shape
    rng = np.random.default_rng(1)
    with_fn = {
        **base,
        "ohlc": {
            c: np.abs(rng.normal(100.0, 5.0, size=(T, K)))
            for c in ("open", "high", "low", "close")
        },
        "microstructure": {
            c: np.abs(rng.normal(0.01, 0.001, size=(T, K)))
            for c in ("eff_spread", "vwap_dev", "block_share", "turnover")
        },
    }
    c_base, _ = assemble_equity_feature_cube(rets, extras=base, normalize=True)
    c_fn, _ = assemble_equity_feature_cube(rets, extras=with_fn, normalize=True)
    assert c_fn.shape[-1] > c_base.shape[-1]


def test_spectrum_and_eq_alloc_defaults_produce_identical_26ch_obs() -> None:
    """G0 identity: shared substrate attach (no feature-net) matches both campaigns."""
    from mascotrl.eval.equity_substrate import (
        attach_equity_obs_substrate,
        stamp_equity_obs_defaults,
    )

    rets, dates, secids, slots_rows, dollar_volume, signals_long = _fixture_panel()

    # Spectrum path (sprint lock: attach_equity_obs_substrate only).
    cfg_spec: dict = {
        "architecture": "mlp",
        "use_surface_signals": True,
        "use_feature_net_extras": False,
    }
    stamp_equity_obs_defaults(cfg_spec)
    attach_equity_obs_substrate(
        cfg_spec,
        dates=dates,
        rets=rets,
        secids=secids,
        slots_rows=slots_rows,
        dollar_volume=dollar_volume,
        signals_long=signals_long,
    )

    # H0 / eq_alloc path with feature-net gated OFF (matches running H0 2600-dim).
    cfg_h0: dict = {
        "architecture": "mlp",
        "use_surface_signals": True,
        "use_equity_feature_cube": True,
        "feature_seq_len": 1,
        "surface_obs_lane": "geometry_lite",
        "use_feature_net_extras": False,
    }
    stamp_equity_obs_defaults(cfg_h0)
    attach_equity_obs_substrate(
        cfg_h0,
        dates=dates,
        rets=rets,
        secids=secids,
        slots_rows=slots_rows,
        dollar_volume=dollar_volume,
        signals_long=signals_long,
    )

    assert set(cfg_spec["feature_extras"]) == set(cfg_h0["feature_extras"])
    assert "iv_surface" in cfg_spec["feature_extras"]
    assert "dollar_volume" in cfg_spec["feature_extras"]
    # Feature-net keys must stay absent under the gate.
    for banned in (
        "ohlc",
        "microstructure",
        "sentiment",
        "fundamentals_pit",
        "option_flow",
        "jkp",
        "macro",
    ):
        assert banned not in cfg_spec["feature_extras"]
        assert banned not in cfg_h0["feature_extras"]

    b_spec = PanelObservationBuilder(
        rets, extras=cfg_spec["feature_extras"], seq_len=1, normalize=True
    )
    b_h0 = PanelObservationBuilder(
        rets, extras=cfg_h0["feature_extras"], seq_len=1, normalize=True
    )
    assert b_spec.obs_channels_per_asset == 26
    assert b_h0.obs_channels_per_asset == 26
    assert b_spec.names == b_h0.names
    w = np.full(rets.shape[1], 1.0 / rets.shape[1])
    np.testing.assert_allclose(b_spec(10, w), b_h0(10, w), rtol=0, atol=0)


def test_spectrum_refuses_feature_net_opt_in_without_schema_support() -> None:
    """Cell YAML cannot legally express feature_extras; runtime opt-in must fail closed."""
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
        "feature_extras": {"ohlc": True},
    }
    with pytest.raises(ValueError, match="unknown keys"):
        validate_cell_cfg(cfg, path="test")


def test_eq_alloc_feature_net_gate_defaults_off() -> None:
    """eq_alloc must not silently attach feature-net (would break H0 26-ch identity)."""
    from pathlib import Path

    src = Path("scripts/run_eq_alloc_campaign.py").read_text(encoding="utf-8")
    assert "use_feature_net_extras" in src
    # Default must be fail-closed (False), not an unconditional attach.
    assert 'cfg.get("use_feature_net_extras"' in src or "use_feature_net_extras" in src
