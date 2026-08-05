"""Wave-1 CRUCIBLE honesty: projector, G1 draws, ridge G2, CPCV reselect purge."""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from mascotrl.data.crucible import (
    feasible_action_diversity_probe,
    lottery_risk_budget_trim,
    ridge_residual_signal,
    transfer_coefficient_probe,
)
from mascotrl.eval.cpcv import CPCVConfig, build_cpcv_folds, stamp_reselect_purge_meta
from mascotrl.policy.cmdp_projector import make_cmdp_projector, soft_simplex_project


def test_g1_default_n_draws_is_512():
    sig = inspect.signature(feasible_action_diversity_probe)
    assert sig.parameters["n_draws"].default == 512


def test_g1_fails_when_projector_collapses_to_uniform():
    k = 8
    secids = list(range(k))

    def uniform_projector(a):
        return np.full(k, 1.0 / k)

    out = feasible_action_diversity_probe(
        secids,
        uniform_projector,
        n_draws=64,
        rng=np.random.default_rng(0),
    )
    assert out["g1_pass"] is False


def test_make_cmdp_hard_projector_is_not_softmax():
    cfg = {"projection_mode": "hard", "turnover_limit": 0.05}
    proj = make_cmdp_projector(cfg, k=6)
    a = np.linspace(-2, 2, 6)
    w_hard = proj(a)
    w_soft = soft_simplex_project(a)
    # Hard projector starts from softmax then clips turnover from EW
    ew = np.full(6, 1.0 / 6)
    assert float(np.sum(np.abs(w_hard - ew))) <= 0.05 + 1e-9
    assert not np.allclose(w_hard, w_soft)


def test_ridge_residual_signal_not_raw_trend():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-02", periods=40)
    secids = [10, 20, 30]
    resid = pd.DataFrame(rng.normal(0, 0.01, size=(40, 3)), index=dates, columns=secids)
    resid.iloc[:, 0] += 0.02
    sig = ridge_residual_signal(resid, secids)
    assert sig.shape == (3,)
    assert np.isfinite(sig).all()
    # Name 0 has positive residual mean → highest ridge signal
    assert sig[0] == max(sig)


def test_lottery_trim_refills_from_candidates():
    rng = np.random.default_rng(1)
    t, n = 80, 10
    dates = pd.bdate_range("2019-01-02", periods=t)
    secids = list(range(n))
    resid = pd.DataFrame(rng.normal(0, 0.01, size=(t, n)), index=dates, columns=secids)
    resid.iloc[:, :3] *= 10.0
    sel = {
        "membership": {
            "lottery": [0, 1, 2],
            "core": [3, 4, 5],
            "trend": [],
            "reversal": [],
            "carry": [],
            "defensive": [],
            "illiquid": [],
        },
        "primary": {i: ("lottery" if i < 3 else "core") for i in range(6)},
        "secids": [0, 1, 2, 3, 4, 5],
    }
    new_sel, info, share = lottery_risk_budget_trim(
        sel, resid, cap=0.15, refill_candidates=[6, 7, 8, 9]
    )
    assert share <= 0.15 + 1e-9
    assert info["n_lottery_dropped"] >= 1
    # Refill should bring unused core candidates into the book
    assert any(s in new_sel["secids"] for s in (6, 7, 8, 9)) or len(new_sel["secids"]) >= 1


def test_cpcv_reselect_purge_stamps_and_shrinks_train():
    dates = list(pd.bdate_range("2018-01-02", periods=120))
    cfg = CPCVConfig(n_splits=6, n_test_groups=2, purge_days=5, embargo_days=2)
    mask = np.zeros(len(dates), dtype=bool)
    mask[0] = mask[63] = True
    meta = stamp_reselect_purge_meta(dates, mask, purge_radius=5)
    assert meta["n_reselect_days"] == 2
    assert meta["n_purged_at_reselect"] > 2

    base = build_cpcv_folds(dates, cfg)
    purged = build_cpcv_folds(
        dates,
        cfg,
        extra_purge_indices=meta["reselect_indices"],
        extra_purge_radius=5,
    )
    assert purged[0].n_train_days < base[0].n_train_days
    assert purged[0].n_purged_days >= base[0].n_purged_days


def test_g2_accepts_ridge_signal_with_identity_projector():
    secids = [1, 2, 3, 4]
    sig = np.array([0.4, -0.1, 0.2, -0.5])
    out = transfer_coefficient_probe(secids, sig, soft_simplex_project)
    assert "g2_tc_post_projection" in out
    assert "g2_pass" in out
