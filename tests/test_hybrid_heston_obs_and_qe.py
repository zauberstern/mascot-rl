"""Hybrid Heston: finetune obs must be finite; Heston default scheme is QE-M."""
from __future__ import annotations

import numpy as np
import pytest
from tests.conftest import FLOAT_TOL

from src.arms import ArmSpec
from src.env.historical_env import HistoricalArmEnv
from src.eval.friction import FrictionSpec


def _raw_env(rets: np.ndarray, factors: np.ndarray) -> HistoricalArmEnv:
    arm = ArmSpec(id="eq", option_slots=0, equity_slots=rets.shape[1], delta_mode="off")
    friction = FrictionSpec(om_touch_enabled=False, equity_bps=5.0)
    return HistoricalArmEnv(
        returns=rets,
        factors=factors,
        arm=arm,
        friction=friction,
        residualizer=None,
    )


def test_historical_env_raw_obs_zero_fills_missing_slots() -> None:
    """Sparse historical panels put NaN in inactive name slots; policy must see finite obs."""
    rets = np.array(
        [[0.01, np.nan, -0.02], [0.0, 0.03, np.nan], [np.nan, np.nan, 0.01]],
        dtype=np.float64,
    )
    fac = np.zeros((3, 4), dtype=np.float64)
    env = _raw_env(rets, fac)
    obs = env._obs()
    assert obs.shape == (3,)
    assert np.isfinite(obs).all()
    assert obs[1] == pytest.approx(0.0, **FLOAT_TOL)# missing slot zero-filled, not leaked as NaN


def test_assert_obs_finite_accepts_hybrid_style_panel_after_env() -> None:
    """Reproduce hybrid_heston fail-closed path: raw NaN returns through env.reset."""
    from src.eval.research_alpha_train import _assert_obs_finite

    rng = np.random.default_rng(0)
    T, K = 16, 8
    rets = rng.normal(0.0, 0.01, size=(T, K))
    rets[:, 3] = np.nan
    rets[5, 1] = np.nan
    fac = rng.normal(0.0, 0.01, size=(T, 4))
    env = _raw_env(rets, fac)
    obs, _info = env.reset()
    clean = _assert_obs_finite(obs, cfg={}, where="policy_obs")
    assert np.isfinite(clean).all()


def test_feller_helper_reports_violation_without_forcing() -> None:
    from src.sim.heston_qe import feller_satisfied, feller_gap

    assert feller_satisfied(kappa=2.0, theta=0.04, xi=0.3)
    assert not feller_satisfied(kappa=1.0, theta=0.04, xi=0.8)
    assert feller_gap(kappa=1.0, theta=0.04, xi=0.8) < 0.0


def test_numpy_qe_m_variance_nonnegative_under_feller_violation() -> None:
    from src.sim.heston_qe import simulate_heston_qe_m

    spots, vars_ = simulate_heston_qe_m(
        n_paths=512,
        n_steps=64,
        dt=1.0 / 252.0,
        spot0=100.0,
        v0=0.04,
        kappa=0.5,
        theta=0.04,
        xi=1.2,
        rho=-0.7,
        rate=0.0,
        div_q=0.0,
        seed=11,
    )
    assert spots.shape == (512, 64)
    assert vars_.shape == (512, 64)
    assert np.isfinite(spots).all()
    assert np.isfinite(vars_).all()
    assert (vars_ >= -1e-12).all()
    assert float(spots.mean()) > 0.0


@pytest.mark.parametrize("scheme", ["qe", "qe_martingale", "full_truncation"])
def test_cpp_heston_scheme_finite(scheme: str) -> None:
    pytest.importorskip("cpp_rbergomi")
    from src.simulator import get_world_bundle
    import torch

    bundle = get_world_bundle(
        {
            "n_paths": 8,
            "n_assets": 2,
            "n_steps": 16,
            "n_strikes": 3,
            "n_maturities": 2,
            "seed": 3,
            "force_world_bundle": True,
            "train_world": "heston",
            "heston_scheme": scheme,
            "heston_kappa": 0.5,
            "heston_theta": 0.04,
            "heston_xi": 1.0,
            "heston_rho": -0.7,
            "heston_v0": 0.04,
        }
    )
    assert torch.isfinite(bundle["spot_paths"]).all()
    assert torch.isfinite(bundle["atm_iv_paths"]).all()
    assert float(bundle["spot_paths"].min()) > 0.0


def test_cpp_heston_default_scheme_is_qe_martingale() -> None:
    pytest.importorskip("cpp_rbergomi")
    import cpp_rbergomi as eng

    w = eng.WorldConfig()
    assert int(w.heston_scheme) == 2  # 0=FT, 1=QE, 2=QE-M


def test_hybrid_fold_allows_cube_false_finetune_with_nan_slots(monkeypatch) -> None:
    """_train_agent_for_fold hybrid path must not die on NaN historical finetune obs."""
    from src.eval import research_alpha_cpcv as mod

    calls: list[str] = []

    def fake_synth(cfg, *, k, n_rows, seed, world):
        calls.append(f"synth:{world}")
        rng = np.random.default_rng(seed)
        return rng.normal(0, 0.01, (n_rows, k)), rng.normal(0, 0.01, (n_rows, 4))

    def fake_train(rets, fac, cfg, seed=0, agent=None):
        phase = "ft" if agent is not None else "pt"
        calls.append(phase)
        from src.eval.research_alpha_train import _assert_obs_finite

        env = _raw_env(rets, fac)
        obs, _ = env.reset()
        _assert_obs_finite(obs, cfg=cfg, where="policy_obs")
        return {"agent": object(), "n_steps": 1, "n_episodes": 1, "mean_reward": 0.0}

    monkeypatch.setattr(mod, "synthetic_train_panel", fake_synth)
    monkeypatch.setattr(mod, "train_research_hist", fake_train)

    T, K = 20, 5
    rets = np.random.default_rng(1).normal(0, 0.01, (T, K))
    rets[:, 2] = np.nan
    fac = np.zeros((T, 4))
    cfg = {
        "train_world": "hybrid_pretrain_finetune",
        "hybrid_pretrain_world": "heston",
        "use_equity_feature_cube": False,
    }
    out = mod._train_agent_for_fold(cfg, rets, fac, np.arange(10), seed=0)
    assert "pretrain_stats" in out
    assert calls[0].startswith("synth:heston")
    assert "pt" in calls and "ft" in calls


def test_quantlib_heston_qe_m_optional_parity() -> None:
    """Optional QuantLib AnalyticHestonEngine + QE-M enum (conformance harness only)."""
    ql = pytest.importorskip("QuantLib")
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    spot = 100.0
    v0, kappa, theta, sigma, rho = 0.04, 2.0, 0.04, 0.3, -0.7
    r, q = 0.01, 0.0
    risk = ql.YieldTermStructureHandle(ql.FlatForward(today, r, ql.Actual365Fixed()))
    div = ql.YieldTermStructureHandle(ql.FlatForward(today, q, ql.Actual365Fixed()))
    assert hasattr(ql.HestonProcess, "QuadraticExponentialMartingale")
    assert hasattr(ql.HestonProcess, "FullTruncation")
    process = ql.HestonProcess(
        risk,
        div,
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        v0,
        kappa,
        theta,
        sigma,
        rho,
        ql.HestonProcess.QuadraticExponentialMartingale,
    )
    model = ql.HestonModel(process)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, 100.0)
    exercise = ql.EuropeanExercise(today + 91)
    opt = ql.VanillaOption(payoff, exercise)
    opt.setPricingEngine(ql.AnalyticHestonEngine(model))
    assert float(opt.NPV()) > 0.0

    pytest.importorskip("cpp_rbergomi")
    from src.simulator import get_world_bundle

    bundle = get_world_bundle(
        {
            "n_paths": 4,
            "n_assets": 1,
            "n_steps": 8,
            "n_strikes": 3,
            "n_maturities": 2,
            "seed": 1,
            "force_world_bundle": True,
            "train_world": "heston",
            "heston_scheme": "qe_martingale",
            "heston_theta": theta,
            "heston_v0": v0,
            "rate": r,
            "div_q": q,
        }
    )
    atm = float(bundle["atm_iv_paths"].mean())
    assert 0.05 < atm < 0.60
