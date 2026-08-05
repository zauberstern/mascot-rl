"""Fail closed when CRUCIBLE community caps cannot fill K."""
from __future__ import annotations

import numpy as np
import pytest

from mascotrl.data.crucible import CrucibleSpec, select_universe_crucible
from mascotrl.eval.friction import FrictionSpec
from mascotrl.policy.cmdp_projector import make_cmdp_projector
from tests.test_crucible_pipeline import _make_rich_panels


def test_community_capacity_must_cover_k():
    """20 communities × 3/community = 60 < K=100 must fail before cryptic projector errors."""
    with pytest.raises(ValueError, match="community capacity"):
        CrucibleSpec(k=100, n_communities=20, max_per_community=3).assert_k_feasible()


def test_projector_adapts_to_selection_size_not_locked_campaign_k():
    """G1 probe uses len(cur_secids); a size-adaptive projector must accept 57 names."""
    cfg = {"projection_mode": "hard", "turnover_limit": 0.15}
    projector = make_cmdp_projector(cfg, k=None)
    raw57 = np.random.default_rng(0).normal(size=57)
    out = np.asarray(projector(raw57), dtype=np.float64)
    assert out.shape == (57,)


def test_locked_projector_still_rejects_size_mismatch():
    cfg = {"projection_mode": "hard", "turnover_limit": 0.15}
    projector = make_cmdp_projector(cfg, k=100)
    with pytest.raises(ValueError, match="projector expected k=100"):
        projector(np.zeros(57))


def test_select_universe_fills_k_when_capacity_allows():
    panels = _make_rich_panels(n=80, t=600, k=20, seed=11)
    quotas = {
        "trend": 4,
        "reversal": 4,
        "carry": 3,
        "defensive": 3,
        "lottery": 2,
        "illiquid": 2,
        "core": 2,
    }
    spec = CrucibleSpec(
        k=20,
        n_communities=10,
        max_per_community=3,
        quotas=quotas,
        g1_l1_floor=0.0,
        g1_entropy_gap_floor=0.0,
        g2_tc_floor=-1.0,
        g3_sharpe_floor=-1.0,
        lottery_resid_var_share_cap=0.99,
    )
    cfg = {"projection_mode": "hard", "turnover_limit": 0.15}
    projector = make_cmdp_projector(cfg, k=None)
    result = select_universe_crucible(
        as_of=panels["returns"].index[-1],
        pool_secids=list(panels["returns"].columns),
        returns=panels["returns"],
        ff4_factors=panels["ff4_factors"],
        adv_panel=panels["adv_panel"],
        amihud_panel=panels["amihud_panel"],
        surface_panel=panels["surface_panel"],
        beta_panel=panels["beta_panel"],
        projector=projector,
        friction_spec=FrictionSpec(equity_bps=5.0, impact_c_eq=0.5),
        spec=spec,
        eligible_secids=list(panels["returns"].columns),
        turnover_limit=0.15,
    )
    assert len(result.secids) == 20
