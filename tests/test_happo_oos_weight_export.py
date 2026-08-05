"""Narrative HAPPO must persist OOS weights for behaviour / desk-org export."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from mascotrl.eval.cpcv import _CPCV_FOLD_AUX_KEY
from mascotrl.eval.research_happo_cpcv import (
    _aggregate_happo_learning_curves,
    _eval_happo_panel_payload,
)


def test_eval_happo_payload_includes_aux_weights_and_turnovers() -> None:
    """Unit shape: eval payload must carry __aux__ with (T,K) weights."""
    dates = list(pd.bdate_range("2020-01-02", periods=8))
    # Synthetic joint book: equal weight drift.
    k = 4
    weights = np.full((6, k), 1.0 / k, dtype=np.float64)
    turnovers = np.linspace(0.01, 0.06, num=6)
    pnl = {str(pd.Timestamp(d).date()): float(i) * 0.001 for i, d in enumerate(dates[:6])}
    payload = _eval_happo_panel_payload(
        pnl=pnl,
        dates=[str(pd.Timestamp(d).date()) for d in dates[:6]],
        weights=weights,
        turnovers=turnovers,
        s_delta=np.zeros(6),
        s_turn=np.zeros(6),
    )
    assert _CPCV_FOLD_AUX_KEY in payload
    aux = payload[_CPCV_FOLD_AUX_KEY]
    assert len(aux) == 6
    first = next(iter(aux.values()))
    assert len(first["weights"]) == k
    assert "turnover" in first
    assert "s_delta" in first and "s_turn" in first
    # PnL keys remain float-convertible for CPCV Sharpe path.
    for key, val in payload.items():
        if key == _CPCV_FOLD_AUX_KEY:
            continue
        assert isinstance(val, float)


def test_aggregate_happo_learning_curves(tmp_path) -> None:
    import json

    curves = tmp_path / "learning_curves"
    curves.mkdir()
    rows = [
        {
            "proj_gap": 0.2,
            "proj_penalty": 0.1,
            "exec_turnover": 0.05,
            "exec_weight_l1": 1.0,
            "teamtr_skips": 1,
            "teamtr_enabled": 1,
            "approx_kl": 0.01,
            "clip_frac": 0.1,
            "entropy": 0.5,
            "agent_order": [0, 1, 2],
            "policy_loss": -0.3,
        },
        {
            "proj_gap": 0.4,
            "proj_penalty": 0.3,
            "exec_turnover": 0.07,
            "exec_weight_l1": 1.0,
            "teamtr_skips": 0,
            "teamtr_enabled": 1,
            "approx_kl": 0.03,
            "clip_frac": 0.2,
            "entropy": 0.7,
            "agent_order": [2, 1, 0],
            "policy_loss": -0.1,
        },
    ]
    (curves / "fold0_seed0_curve.json").write_text(
        json.dumps(rows) + "\n", encoding="utf-8"
    )
    agg = _aggregate_happo_learning_curves(curves)
    assert agg["proj_gap_mean"] == pytest.approx(0.3)
    assert agg["teamtr_skips_sum"] == pytest.approx(1.0)
    assert agg["policy_loss_last_agent_only"] is True
    assert agg["agent_order_entropy"] is not None
    assert np.isfinite(float(agg["agent_order_entropy"]))


def test_run_happo_cpcv_toy_emits_weights(monkeypatch, tmp_path) -> None:
    """Integration: narrative toy CPCV artifact must hoistable weights."""
    from mascotrl.eval import research_happo_cpcv as rh

    def _fake_train(train_rets, cfg, *, arm, seed):
        class _Pol:
            pass

        return _Pol(), [0.05], [{"proj_gap": 0.1, "teamtr_skips": 0, "entropy": 0.2}]

    def _fake_eval(eq_rets, test_dates, cfg, *, arm, policy):
        k_assets = int(np.asarray(eq_rets).shape[1])
        n = min(5, len(test_dates))
        dates = [str(pd.Timestamp(d).date()) for d in test_dates[:n]]
        w = np.full((n, k_assets), 1.0 / k_assets, dtype=np.float64)
        turns = np.full(n, 0.05, dtype=np.float64)
        pnl = {d: 0.001 for d in dates}
        return rh._eval_happo_panel_payload(
            pnl=pnl,
            dates=dates,
            weights=w,
            turnovers=turns,
            s_delta=np.zeros(n),
            s_turn=np.zeros(n),
        )

    monkeypatch.setattr(rh, "_train_happo_on_panel", _fake_train)
    monkeypatch.setattr(rh, "_eval_happo_on_panel", _fake_eval)

    cfg: dict[str, Any] = {
        "n_assets": 4,
        "d_model": 8,
        "macro_dim": 4,
        "seed": 0,
        "protocol_tier": "narrative",
        "claim_tier": "research",
        "execution_spread_bps": 5.0,
        "cpcv_purge_days": 0,
        "cpcv_embargo_days": 0,
        "train_env_steps": 8,
    }
    budget = {
        "seeds": [0],
        "cpcv_n_splits": 2,
        "cpcv_n_test_groups": 1,
        "claim_tier": "research",
        "n_episodes": 2,
        "horizon": 8,
        "dispatch_only": False,
    }
    art, err = rh.run_happo_cpcv(
        cfg,
        "eq",
        budget=budget,
        allow_toy_panel=True,
        no_dry_run=False,
        out_dir=tmp_path / "happo_out",
        resume=False,
    )
    assert err is None, err
    assert art is not None
    assert "weights" in art or (art.get("paths") or {}).get("0", {}).get("weights")
    w = art.get("weights") or art["paths"]["0"]["weights"]
    arr = np.asarray(w, dtype=np.float64)
    assert arr.ndim == 2 and arr.shape[1] == 4
    assert arr.shape[0] >= 1
    assert "happo_trainer_stats" in art or "coordination_proxies" in art
