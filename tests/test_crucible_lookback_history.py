"""CRUCIBLE must residualise with pre-eval history (not eval-only returns)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mascotrl.data.crucible import CrucibleSpec, residual_communities, select_universe_crucible
from mascotrl.eval.friction import FrictionSpec


def _softmax_project(a):
    a = np.asarray(a, dtype=np.float64).reshape(-1) * 6.0
    z = a - np.max(a)
    e = np.exp(z)
    return e / np.maximum(e.sum(), 1e-12)


def _panels_eval_only(*, n: int = 40, k: int = 12, seed: int = 0):
    """Simulate campaign bug: returns start on as_of with no lookback history."""
    rng = np.random.default_rng(seed)
    as_of = pd.Timestamp("2014-01-02")
    # Only a handful of eval days — residual window cannot warm up
    dates = pd.bdate_range(as_of, periods=5)
    secids = list(range(1, n + 1))
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(len(dates), n)), index=dates, columns=secids)
    ff4 = pd.DataFrame(
        rng.normal(0, 0.01, size=(len(dates), 4)),
        index=dates,
        columns=["mkt_rf", "smb", "hml", "umd"],
    )
    # Varied ADV so C1 strata can fill (constant ADV collapses to one band)
    adv = pd.DataFrame(
        rng.uniform(5e6, 5e8, size=(len(dates), n)), index=dates, columns=secids
    )
    amihud = rets.abs() / adv
    beta = pd.DataFrame(1.0, index=dates, columns=secids)
    surface = pd.DataFrame(
        {
            "date": np.repeat(dates, n),
            "secid": np.tile(secids, len(dates)),
            "signal": "mfis_30",
            "value": 0.2,
        }
    )
    return {
        "as_of": as_of,
        "pool_secids": secids,
        "returns": rets,
        "ff4_factors": ff4,
        "adv_panel": adv,
        "amihud_panel": amihud,
        "surface_panel": surface,
        "beta_panel": beta,
        "projector": _softmax_project,
        "friction_spec": FrictionSpec(),
        "spec": CrucibleSpec(
            k=k,
            max_pool=n,
            lookback_days=252,
            g1_l1_floor=0.01,
            g1_entropy_gap_floor=0.1,
            g2_tc_floor=0.0,
            g3_sharpe_floor=-10.0,
            lottery_resid_var_share_cap=0.99,
            n_communities=8,
            max_per_community=5,
        ),
        "eligible_secids": secids,
        "rng_seed": seed,
    }


def _panels_with_history(*, n: int = 40, k: int = 12, seed: int = 0):
    """Same as_of but with 600 business days of history before eval."""
    rng = np.random.default_rng(seed)
    as_of = pd.Timestamp("2014-01-02")
    hist = pd.bdate_range(as_of - pd.offsets.BDay(600), as_of - pd.offsets.BDay(1))
    eval_dates = pd.bdate_range(as_of, periods=5)
    dates = hist.append(eval_dates)
    secids = list(range(1, n + 1))
    factors = rng.normal(0, 0.01, size=(len(dates), 6))
    loads = rng.normal(0, 1.0, size=(n, 6))
    idio = rng.normal(0, 0.012, size=(len(dates), n))
    rets = factors @ loads.T + idio
    returns = pd.DataFrame(rets, index=dates, columns=secids)
    ff4 = pd.DataFrame(
        rng.normal(0, 0.01, size=(len(dates), 4)),
        index=dates,
        columns=["mkt_rf", "smb", "hml", "umd"],
    )
    adv = pd.DataFrame(
        rng.uniform(5e6, 5e8, size=(len(dates), n)), index=dates, columns=secids
    )
    amihud = returns.abs() / adv
    beta = pd.DataFrame(1.0, index=dates, columns=secids)
    # Surface only needs recent coverage for eligibility
    surf_dates = dates[-80:]
    surface = pd.DataFrame(
        {
            "date": np.repeat(surf_dates, n),
            "secid": np.tile(secids, len(surf_dates)),
            "signal": "mfis_30",
            "value": 0.2,
        }
    )
    return {
        "as_of": as_of,
        "pool_secids": secids,
        "returns": returns,
        "ff4_factors": ff4,
        "adv_panel": adv,
        "amihud_panel": amihud,
        "surface_panel": surface,
        "beta_panel": beta,
        "projector": _softmax_project,
        "friction_spec": FrictionSpec(),
        "spec": CrucibleSpec(
            k=k,
            max_pool=n,
            lookback_days=252,
            g1_l1_floor=0.01,
            g1_entropy_gap_floor=0.1,
            g2_tc_floor=0.0,
            g3_sharpe_floor=-10.0,
            lottery_resid_var_share_cap=0.99,
            n_communities=8,
            max_per_community=5,
        ),
        "eligible_secids": secids,
        "rng_seed": seed,
    }


def test_eval_only_returns_fail_closed_with_clear_lookback_error():
    kw = _panels_eval_only()
    with pytest.raises(ValueError, match="lookback too short|need at least 2 names"):
        select_universe_crucible(**kw)


def test_pre_eval_history_allows_first_as_of_selection():
    kw = _panels_with_history()
    result = select_universe_crucible(**kw)
    assert len(result.secids) == kw["spec"].k
    assert result.fingerprint


def test_residual_communities_all_nan_raises():
    dates = pd.bdate_range("2014-01-02", periods=10)
    resid = pd.DataFrame(np.nan, index=dates, columns=[1, 2, 3])
    with pytest.raises(ValueError, match="need at least 2 names"):
        residual_communities(resid, n_communities=3, min_n_eff_enb=None)
