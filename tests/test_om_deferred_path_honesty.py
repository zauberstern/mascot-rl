"""A-7/A-8: OM deferred-path honesty stamps and claim-tier refusal."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.oos_panel import extract_om_marks


def _mix_panel_df(*, T: int = 4, n_opt: int = 2, n_eq: int = 3) -> pd.DataFrame:
    """Wide panel with opt columns; equity spot_2 missing to force KeyError."""
    rng = np.random.default_rng(0)
    data: dict[str, np.ndarray] = {}
    for i in range(n_opt):
        data[f"bid_ask_spread_{i}"] = rng.uniform(0.01, 0.02, T)
        data[f"delta_{i}"] = rng.uniform(0.3, 0.6, T)
        data[f"spot_{i}"] = rng.uniform(90.0, 110.0, T)
        data[f"dh_denom_{i}"] = rng.uniform(40.0, 60.0, T)
        data[f"dh_ret_{i}"] = rng.normal(0.0, 0.01, T)
    # Equity block reuses spot_0..spot_{n_eq-1}; omit spot_2 so eq lookup degrades.
    for i in range(n_eq):
        if i >= 2:
            continue
        if f"spot_{i}" not in data:
            data[f"spot_{i}"] = rng.uniform(50.0, 60.0, T)
    idx = pd.bdate_range("2020-01-01", periods=T)
    return pd.DataFrame(data, index=idx)


def test_extract_om_marks_stamps_degraded_on_equity_spot_keyerror() -> None:
    """A-8: missing equity spot columns must stamp om_marks_degraded."""
    df = _mix_panel_df(T=5, n_opt=2, n_eq=3)
    marks = extract_om_marks(df, n_opt=2, n_eq=3, label_stem="dh_ret")
    assert marks.get("om_marks_degraded") is True
    assert marks["spot"].shape == (5, 5)


def test_try_load_om_panel_stamps_pit_as_of_and_factors_source() -> None:
    """A-7: deferred OM load must stamp pit_as_of and factors_source on cfg."""
    from scripts.run_research_alpha_cpcv import _try_load_om_panel

    T, k = 6, 2
    panel = _mix_panel_df(T=T, n_opt=k, n_eq=0)
    rets = np.zeros((T, k), dtype=np.float64)

    cfg = {
        "portfolio_arm": "opt",
        "claim_tier": "research",
        "n_assets": k,
    }
    mock_arm = MagicMock()
    mock_arm.id = "opt"
    mock_arm.option_label_stem = "dh_ret"

    with (
        patch("src.data.arctic_store.ArcticStateStore", return_value=MagicMock()),
        patch("src.data.oos_panel.load_oos_panel", return_value=(panel, [1, 2])),
        patch("src.data.oos_panel.label_matrix", return_value=rets),
        patch("src.data.oos_panel.extract_om_marks", return_value={"half_spread": rets}),
        patch("src.arms.training.resolve_portfolio_arm", return_value=mock_arm),
        patch("src.arms.training.resolve_claim_label_stem", return_value="dh_ret"),
        patch("src.eval.equity_factors._try_load_ff_panel", return_value=None),
    ):
        loaded = _try_load_om_panel(cfg, k)

    assert loaded is not None
    assert "pit_as_of" in cfg
    assert cfg["pit_as_of"] is None
    assert cfg.get("factors_source") == "zeros"


def test_try_load_om_panel_stamps_factors_source_ff4() -> None:
    from scripts.run_research_alpha_cpcv import _try_load_om_panel

    T, k = 6, 2
    panel = _mix_panel_df(T=T, n_opt=k, n_eq=0)
    rets = np.zeros((T, k), dtype=np.float64)
    ff = pd.DataFrame(
        np.zeros((T, 4)),
        index=panel.index,
        columns=["mkt", "smb", "hml", "mom"],
    )
    cfg = {"portfolio_arm": "opt", "claim_tier": "screening", "n_assets": k}
    mock_arm = MagicMock()
    mock_arm.id = "opt"
    mock_arm.option_label_stem = "dh_ret"

    with (
        patch("src.data.arctic_store.ArcticStateStore", return_value=MagicMock()),
        patch("src.data.oos_panel.load_oos_panel", return_value=(panel, [1, 2])),
        patch("src.data.oos_panel.label_matrix", return_value=rets),
        patch("src.data.oos_panel.extract_om_marks", return_value={"half_spread": rets}),
        patch("src.arms.training.resolve_portfolio_arm", return_value=mock_arm),
        patch("src.arms.training.resolve_claim_label_stem", return_value="dh_ret"),
        patch("src.eval.equity_factors._try_load_ff_panel", return_value=ff),
    ):
        _try_load_om_panel(cfg, k)

    assert cfg.get("factors_source") == "ff4"


def test_try_load_om_panel_allows_dispatch_only_for_opt() -> None:
    """A-7: HAPPO screening smoke claim_tier=dispatch_only must remain legal."""
    from scripts.run_research_alpha_cpcv import _try_load_om_panel

    T, k = 6, 2
    panel = _mix_panel_df(T=T, n_opt=k, n_eq=0)
    rets = np.zeros((T, k), dtype=np.float64)
    cfg = {
        "portfolio_arm": "opt",
        "claim_tier": "dispatch_only",
        "n_assets": k,
    }
    mock_arm = MagicMock()
    mock_arm.id = "opt"
    mock_arm.option_label_stem = "dh_ret"

    with (
        patch("src.data.arctic_store.ArcticStateStore", return_value=MagicMock()),
        patch("src.data.oos_panel.load_oos_panel", return_value=(panel, [1, 2])),
        patch("src.data.oos_panel.label_matrix", return_value=rets),
        patch("src.data.oos_panel.extract_om_marks", return_value={"half_spread": rets}),
        patch("src.arms.training.resolve_portfolio_arm", return_value=mock_arm),
        patch("src.arms.training.resolve_claim_label_stem", return_value="dh_ret"),
        patch("src.eval.equity_factors._try_load_ff_panel", return_value=None),
    ):
        loaded = _try_load_om_panel(cfg, k)
    assert loaded is not None


def test_try_load_om_panel_refuses_capital_claim_tier_for_opt() -> None:
    """A-7: opt/mix deferred path must refuse capital-grade claim tiers."""
    from scripts.run_research_alpha_cpcv import _try_load_om_panel

    T, k = 6, 2
    panel = _mix_panel_df(T=T, n_opt=k, n_eq=0)
    rets = np.zeros((T, k), dtype=np.float64)
    cfg = {
        "portfolio_arm": "opt",
        "claim_tier": "narrative",
        "n_assets": k,
    }
    mock_arm = MagicMock()
    mock_arm.id = "opt"
    mock_arm.option_label_stem = "dh_ret"

    with (
        patch("src.data.arctic_store.ArcticStateStore", return_value=MagicMock()),
        patch("src.data.oos_panel.load_oos_panel", return_value=(panel, [1, 2])),
        patch("src.data.oos_panel.label_matrix", return_value=rets),
        patch("src.data.oos_panel.extract_om_marks", return_value={"half_spread": rets}),
        patch("src.arms.training.resolve_portfolio_arm", return_value=mock_arm),
        patch("src.arms.training.resolve_claim_label_stem", return_value="dh_ret"),
        patch("src.eval.equity_factors._try_load_ff_panel", return_value=None),
    ):
        with pytest.raises(ValueError, match="rematerialize|claim_tier"):
            _try_load_om_panel(cfg, k)


def test_run_research_alpha_cpcv_propagates_om_honesty_stamps() -> None:
    """A-7: cfg honesty stamps must appear on CPCV artifacts."""
    from src.eval.cpcv import CPCVConfig
    from src.eval.research_alpha_cpcv import run_research_alpha_cpcv

    t, k = 90, 4
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0003, 0.01, size=(t, k))
    factors = rng.normal(0.0, 0.005, size=(t, 4))
    dates = pd.bdate_range("2020-01-01", periods=t)
    cfg = {
        "claim_tier": "research",
        "primary_train": "historical_arm_env",
        "portfolio_arm": "eq",
        "headline_fill": "pct75",
        "n_assets": k,
        "train_epochs": 1,
        "lr": 3e-4,
        "pit_as_of": None,
        "factors_source": "zeros",
        "om_marks_degraded": True,
    }
    cpcv = CPCVConfig(n_splits=3, n_test_groups=1, purge_days=0, embargo_days=0)
    art = run_research_alpha_cpcv(
        dates, rets, factors, cfg, cpcv=cpcv, seed=0, panel_source="optionmetrics"
    )
    assert art.get("pit_as_of") is None
    assert art.get("factors_source") == "zeros"
    assert art.get("om_marks_degraded") is True
