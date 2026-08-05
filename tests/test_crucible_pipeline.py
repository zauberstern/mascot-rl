"""CRUCIBLE end-to-end pipeline stages C0-C13."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mascotrl.data.crucible import (
    SLEEVE_QUOTAS,
    CrucibleGateFailure,
    CrucibleSpec,
    attrition_funnel_report,
    lottery_risk_budget_trim,
    pack_slots_by_community,
    residual_communities,
    select_universe_crucible,
)
from mascotrl.eval.friction import FrictionSpec


def _softmax_project(a):
    a = np.asarray(a, dtype=np.float64).reshape(-1) * 6.0
    z = a - np.max(a)
    e = np.exp(z)
    return e / np.maximum(e.sum(), 1e-12)


def _make_rich_panels(*, n: int = 48, t: int = 600, k: int = 20, seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-02", periods=t)
    secids = list(range(1, n + 1))
    # Low-corr residual drivers so n_eff_enb stays healthy
    factors = rng.normal(0, 0.01, size=(t, 6))
    loads = rng.normal(0, 1.0, size=(n, 6))
    idio = rng.normal(0, 0.012, size=(t, n))
    rets = factors @ loads.T + idio
    # Plant a cross-sectional signal so G3 ridge can clear non-negativity rungs
    signal = rng.normal(0, 1.0, size=n)
    rets = rets + 0.002 * signal.reshape(1, -1)
    returns = pd.DataFrame(rets, index=dates, columns=secids)
    ff4 = pd.DataFrame(
        rng.normal(0, 0.01, size=(t, 4)),
        index=dates,
        columns=["mkt_rf", "smb", "hml", "umd"],
    )
    # ADV strata: high / mid / low thirds
    adv_levels = np.concatenate(
        [
            np.full(n // 3, 8e7),
            np.full(n // 3, 2e7),
            np.full(n - 2 * (n // 3), 5e6),
        ]
    )
    adv = pd.DataFrame(
        np.tile(adv_levels, (t, 1)) * rng.uniform(0.9, 1.1, size=(t, n)),
        index=dates,
        columns=secids,
    )
    amihud = pd.DataFrame(
        rng.uniform(1e-8, 1e-5, size=(t, n)), index=dates, columns=secids
    )
    beta = pd.DataFrame(
        rng.uniform(0.5, 1.5, size=(t, n)), index=dates, columns=secids
    )
    rows = []
    for j, sid in enumerate(secids):
        for d in dates:
            rows.append(
                {
                    "date": d,
                    "secid": sid,
                    "signal": "mfis_30",
                    "value": float(rng.normal()),
                }
            )
            rows.append(
                {
                    "date": d,
                    "secid": sid,
                    "signal": "mfis_365",
                    "value": float(rng.normal()),
                }
            )
            rows.append(
                {
                    "date": d,
                    "secid": sid,
                    "signal": "hv_20",
                    "value": 0.15 + 0.01 * (j % 5),
                }
            )
            rows.append(
                {
                    "date": d,
                    "secid": sid,
                    "signal": "iv_30",
                    "value": 0.18 + 0.01 * (j % 7),
                }
            )
            rows.append(
                {
                    "date": d,
                    "secid": sid,
                    "signal": "skew",
                    "value": float(rng.normal(0, 0.15)),
                }
            )
            rows.append(
                {
                    "date": d,
                    "secid": sid,
                    "signal": "term",
                    "value": float(rng.normal(0, 0.08)),
                }
            )
    surface = pd.DataFrame(rows)
    # Scale sleeve quotas to k (preserve relative shares of SLEEVE_QUOTAS)
    raw = dict(SLEEVE_QUOTAS)
    total = sum(raw.values())
    quotas = {s: max(1, int(round(k * raw[s] / total))) for s in raw}
    # Fix rounding so sum == k
    while sum(quotas.values()) > k:
        for s in ("core", "illiquid", "lottery"):
            if quotas[s] > 1 and sum(quotas.values()) > k:
                quotas[s] -= 1
    while sum(quotas.values()) < k:
        quotas["core"] += 1
    spec = CrucibleSpec(
        k=k,
        max_pool=n,
        lookback_days=252,
        n_communities=8,
        max_per_community=4,
        quotas=quotas,
        g1_l1_floor=0.05,
        g1_entropy_gap_floor=0.3,
        g2_tc_floor=0.05,
        g3_sharpe_floor=-10.0,  # disable G3 for pipeline smoke
        max_repair_passes=5,
        lottery_resid_var_share_cap=0.99,
    )
    return {
        "as_of": dates[-1],
        "pool_secids": secids,
        "returns": returns,
        "ff4_factors": ff4,
        "adv_panel": adv,
        "amihud_panel": amihud,
        "surface_panel": surface,
        "beta_panel": beta,
        "projector": _softmax_project,
        "friction_spec": FrictionSpec(equity_bps=0.0, impact_c_eq=0.0, execution_impact_coef=0.0),
        "spec": spec,
        "rng_seed": 0,
        "book_notional": 1_000_000.0,
    }


def test_attrition_funnel_monotone_and_sums():
    stages = [("parent", 100), ("adv", 80), ("amihud", 70), ("option", 60)]
    rep = attrition_funnel_report(stages)
    funnel = rep["attrition_funnel"]
    assert funnel[0]["n_in"] == 100
    assert funnel[-1]["n_out"] == 60
    for row in funnel:
        assert row["n_out"] == row["n_in"] - row["n_dropped"]
        assert row["n_out"] <= row["n_in"]


def test_fail_closed_eligible_below_2k():
    kw = _make_rich_panels(n=30, k=20)
    # Empty surface + no eligible override => option stage empties pool
    kw["surface_panel"] = pd.DataFrame(columns=["date", "secid", "signal", "value"])
    with pytest.raises(ValueError, match="option|2 \\* K|eligible"):
        select_universe_crucible(**kw)


def test_fail_closed_empty_adv_stratum():
    kw = _make_rich_panels(n=48, k=20)
    # Collapse ADV so floor fails for all names
    kw["adv_panel"].loc[:, :] = 1.0
    kw["book_notional"] = 1e12
    with pytest.raises(ValueError, match="adv|stratum|floor"):
        select_universe_crucible(**kw)


def test_fail_closed_n_eff_enb():
    n, t, k = 40, 280, 14
    dates = pd.bdate_range("2016-01-04", periods=t)
    secids = list(range(n))
    # Rank-1 returns => tiny ENB
    common = np.random.default_rng(0).normal(0, 0.02, size=(t, 1))
    rets = pd.DataFrame(common @ np.ones((1, n)), index=dates, columns=secids)
    resid = rets  # community on residuals of identical columns
    with pytest.raises(ValueError, match="n_eff_enb"):
        residual_communities(resid, n_communities=10, min_n_eff_enb=12.0)


def test_lottery_risk_budget_trims():
    rng = np.random.default_rng(1)
    t, n = 100, 12
    dates = pd.bdate_range("2019-01-02", periods=t)
    secids = list(range(n))
    resid = pd.DataFrame(rng.normal(0, 0.01, size=(t, n)), index=dates, columns=secids)
    # Make lottery names dominate residual variance
    resid.iloc[:, :3] *= 8.0
    sel = {
        "membership": {
            "lottery": [0, 1, 2],
            "core": [3, 4, 5, 6, 7, 8, 9, 10, 11],
            "trend": [],
            "reversal": [],
            "carry": [],
            "defensive": [],
            "illiquid": [],
        },
        "primary": {i: ("lottery" if i < 3 else "core") for i in secids},
        "secids": secids,
    }
    new_sel, info, share = lottery_risk_budget_trim(sel, resid, cap=0.20)
    assert share <= 0.20 + 1e-9
    assert info["lottery_resid_var_share_pre"] > info["lottery_resid_var_share_post"]
    assert sum(1 for s in new_sel["primary"].values() if s == "lottery") < 3


def test_community_cap_never_exceeded():
    kw = _make_rich_panels(k=20, n=48)
    kw["spec"] = CrucibleSpec(
        **{
            **kw["spec"].__dict__,
            "max_per_community": 3,
            "g3_sharpe_floor": -10.0,
            "lottery_resid_var_share_cap": 0.99,
            "g1_l1_floor": 0.01,
            "g1_entropy_gap_floor": 0.1,
            "g2_tc_floor": 0.0,
        }
    )
    result = select_universe_crucible(**kw)
    from collections import Counter

    c = Counter(result.community_of[s] for s in result.secids)
    assert max(c.values()) <= 3


def test_repair_stops_and_raises_gate_failure():
    kw = _make_rich_panels(k=20, n=48)
    # Impossible G1 floor with uniform projector
    kw["projector"] = lambda a: np.full(np.asarray(a).size, 1.0 / np.asarray(a).size)
    kw["spec"] = CrucibleSpec(
        **{
            **kw["spec"].__dict__,
            "g1_l1_floor": 0.5,
            "g1_entropy_gap_floor": 2.0,
            "g2_tc_floor": 0.0,
            "g3_sharpe_floor": -10.0,
            "max_repair_passes": 5,
            "lottery_resid_var_share_cap": 0.99,
        }
    )
    with pytest.raises(CrucibleGateFailure) as ei:
        select_universe_crucible(**kw)
    assert ei.value.diagnostics is not None
    assert ei.value.diagnostics.get("repair_passes_used", 0) <= 5


def test_output_deterministic_for_fixed_seed():
    kw = _make_rich_panels(k=20, n=48, seed=11)
    kw["spec"] = CrucibleSpec(
        **{
            **kw["spec"].__dict__,
            "g3_sharpe_floor": -10.0,
            "g1_l1_floor": 0.01,
            "g1_entropy_gap_floor": 0.1,
            "g2_tc_floor": 0.0,
            "lottery_resid_var_share_cap": 0.99,
        }
    )
    a = select_universe_crucible(**kw)
    b = select_universe_crucible(**kw)
    assert a.secids == b.secids
    assert a.fingerprint == b.fingerprint


def test_partition_scores_group_communities_contiguously():
    secids = [10, 11, 20, 21, 30]
    community_of = {10: 2, 11: 2, 20: 0, 21: 0, 30: 1}
    ordered, scores = pack_slots_by_community(secids, community_of)
    # Communities appear as contiguous blocks
    seen = []
    for s in ordered:
        c = community_of[s]
        if not seen or seen[-1] != c:
            seen.append(c)
        else:
            assert seen[-1] == c
    assert len(seen) == len(set(community_of[s] for s in ordered))
    assert len(scores) == len(ordered)
