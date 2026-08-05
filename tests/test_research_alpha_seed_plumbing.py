"""TDD (W2.2): seed must be set before agent construction, and env.reset(seed=...) must not raise."""
from __future__ import annotations

import numpy as np

from mascotrl.env.historical_env import HistoricalArmEnv
from mascotrl.eval.research_alpha_train import train_research_hist


def _toy_panel(t: int = 30, k: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, 0.01, size=(t, k))
    factors = rng.normal(0.0, 0.01, size=(t, 4))
    return rets, factors


def _flatten_state_dict(agent) -> np.ndarray:
    parts = [p.detach().cpu().numpy().reshape(-1) for p in agent.net.parameters()]
    return np.concatenate(parts)


def test_different_seeds_before_init_yield_different_initial_params() -> None:
    rets, fac = _toy_panel()
    cfg = {
        "primary_train": "historical_arm_env", "portfolio_arm": "eq",
        "n_assets": 3,
        "train_epochs": 0,
        "train_episodes": 0,
        "train_env_steps": 0,
        "policy": "single_agent",
        "projection_mode": "soft",
        "rl_backend": "custom",
    }
    out1 = train_research_hist(rets, fac, dict(cfg), seed=1)
    out2 = train_research_hist(rets, fac, dict(cfg), seed=2)
    v1 = _flatten_state_dict(out1["agent"])
    v2 = _flatten_state_dict(out2["agent"])
    assert v1.shape == v2.shape
    assert not np.allclose(v1, v2)


def test_historical_arm_env_reset_with_seed_does_not_raise() -> None:
    from mascotrl.arms import ArmSpec
    from mascotrl.eval.friction import FrictionSpec

    rets, fac = _toy_panel(t=20, k=2)
    arm = ArmSpec(id='eq', option_slots=0, equity_slots=2, delta_mode='off')
    friction = FrictionSpec(om_touch_enabled=False)
    env = HistoricalArmEnv(returns=rets, factors=fac, arm=arm, friction=friction, residualizer=None)
    obs, info = env.reset(seed=1)
    assert obs is not None
    assert env.t == 1
