"""HistoricalArmEnv contract: finite obs, determinism, simplex, termination."""
from __future__ import annotations

import numpy as np
import pytest

from src.arms import ArmSpec
from src.env.historical_env import HistoricalArmEnv
from src.eval.friction import FrictionSpec
from src.eval.residualization import fit_ff4_residualizer, freeze_residualizer
from tests.conftest import FLOAT_TOL


def _toy_env(T: int = 50, K: int = 5, seed: int = 42) -> HistoricalArmEnv:
    rng = np.random.default_rng(seed)
    rets = rng.normal(scale=0.01, size=(T, K))
    factors = rng.normal(scale=0.005, size=(T, 4))
    arm = ArmSpec(id="eq", option_slots=0, equity_slots=K, delta_mode="off")
    friction = FrictionSpec(
        spec_id="v2_quote_touch", equity_bps=5.0, om_touch_enabled=False
    )
    y = rets.mean(axis=1)
    resid = freeze_residualizer(fit_ff4_residualizer(y, factors, fold_id="f0"), "f0")
    return HistoricalArmEnv(
        returns=rets,
        factors=factors,
        arm=arm,
        friction=friction,
        residualizer=resid,
    )


@pytest.mark.integration
def test_env_reset_finite_obs(torch_deterministic):
    """reset() must return finite, correctly-shaped observations."""
    env = _toy_env(T=50, K=5)
    obs, info = env.reset()
    assert np.all(np.isfinite(obs))
    assert obs.shape[-1] > 0
    assert isinstance(info, dict)


@pytest.mark.integration
def test_env_step_determinism(torch_deterministic):
    """Same seed + same action sequence = identical reward + next_obs."""
    results = []
    for _ in range(2):
        env = _toy_env(T=50, K=5, seed=42)
        obs, _ = env.reset()
        w = np.full(5, 0.2)
        obs2, r, term, trunc, info = env.step(w)
        results.append((float(r), obs2.copy(), bool(term), bool(trunc)))
    r0, obs0, t0, tr0 = results[0]
    r1, obs1, t1, tr1 = results[1]
    assert r1 == pytest.approx(r0, abs=1e-12)
    np.testing.assert_array_equal(obs1, obs0)
    assert (t0, tr0) == (t1, tr1)


@pytest.mark.integration
def test_env_weights_sum_after_project():
    """Post-step info weights (if present) or held portfolio stay finite."""
    env = _toy_env(T=30, K=5)
    env.reset()
    w = np.array([0.4, 0.3, 0.2, 0.1, 0.0])
    assert float(w.sum()) == pytest.approx(1.0, **FLOAT_TOL)
    _, r, _, _, info = env.step(w)
    assert np.isfinite(r)
    for key in ("gross", "cost", "residual"):
        if key in info:
            assert np.isfinite(info[key])


@pytest.mark.integration
def test_env_episode_terminates():
    """Episode must terminate/truncate after T steps, not hang."""
    T, K = 20, 4
    env = _toy_env(T=T, K=K)
    env.reset()
    w = np.full(K, 1.0 / K)
    done = False
    steps = 0
    for _ in range(T + 5):
        _, _, term, trunc, _ = env.step(w)
        steps += 1
        if term or trunc:
            done = True
            break
    assert done, f"env did not terminate after {steps} steps (T={T})"
    assert steps <= T
