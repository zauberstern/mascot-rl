"""RC2: policy_mode risk-aversion must reach episode_weights in research train."""
from __future__ import annotations

import pytest

from mascotrl.spectrum.policy_mode import apply_risk_aversion, risk_aversion_multiplier


@pytest.mark.parametrize(
    "mode,base,expected",
    [
        ("archetype_carry", 1.5, 0.75),
        ("archetype_crisis", 1.5, 3.0),
        ("archetype_inflation", 1.5, 1.5),  # identity without term_spread_z
        ("shared", 1.5, 1.5),
    ],
)
def test_apply_risk_aversion_scales(mode: str, base: float, expected: float) -> None:
    scaled = apply_risk_aversion(base, mode)
    assert abs(scaled - expected) < 1e-9


def test_archetype_inflation_scales_with_term_spread_z() -> None:
    scaled = apply_risk_aversion(1.5, "archetype_inflation", term_spread_z=2.0)
    assert abs(scaled - 1.5 * 1.5) < 1e-9  # 1.5 * (1 + 0.25*2)


def test_research_train_scales_cao_c_for_carry(monkeypatch) -> None:
    """archetype_carry must pass cao_c*0.5 into episode_weights."""
    import numpy as np

    import mascotrl.eval.research_alpha_train as rat
    from mascotrl.policy.objective_factory import episode_weights as real_ew

    captured: list[float] = []

    def spy_ew(mode, G, *, cao_c=1.5, kappa=1.0, **kwargs):
        captured.append(float(cao_c))
        return real_ew(mode, G, cao_c=cao_c, kappa=kappa, **kwargs)

    monkeypatch.setattr(rat, "episode_weights", spy_ew)

    # Minimal synthetic panel: enough days for one short episode.
    T, K = 32, 4
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.01, size=(T, K)).astype(np.float64)
    factors = rng.normal(0.0, 0.01, size=(T, 4)).astype(np.float64)
    cfg = {
        "algo": "ppo",
        "architecture": "mlp",
        "objective": "mean_std_cao",
        "objective_primary": True,
        "primary_train": "historical_arm_env",
        "portfolio_arm": "eq",
        "om_touch_enabled": True,
        "hedge_leg_spread_bps": 5.0,
        "policy_mode": "archetype_carry",
        "cao_c": 1.5,
        "kappa": 1.0,
        "train_env_steps": 16,
        "train_episodes": 1,
        "train_epochs": 1,
        "n_minibatches": 1,
        "weight_head": "softmax",
        "n_assets": K,
        "projection_mode": "soft",
        "claim_tier": "research",
    }
    out = rat.train_research_hist(returns, factors, cfg, seed=0)
    assert out is not None
    assert captured, "episode_weights was never called"
    assert abs(captured[0] - 0.75) < 1e-9, (
        f"expected cao_c=0.75 for archetype_carry, got {captured[0]}"
    )


def test_research_train_scales_cao_c_for_crisis(monkeypatch) -> None:
    import numpy as np

    import mascotrl.eval.research_alpha_train as rat
    from mascotrl.policy.objective_factory import episode_weights as real_ew

    captured: list[float] = []

    def spy_ew(mode, G, *, cao_c=1.5, kappa=1.0, **kwargs):
        captured.append(float(cao_c))
        return real_ew(mode, G, cao_c=cao_c, kappa=kappa, **kwargs)

    monkeypatch.setattr(rat, "episode_weights", spy_ew)

    T, K = 32, 4
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0, 0.01, size=(T, K)).astype(np.float64)
    factors = rng.normal(0.0, 0.01, size=(T, 4)).astype(np.float64)
    cfg = {
        "algo": "ppo",
        "architecture": "mlp",
        "objective": "mean_std_cao",
        "objective_primary": True,
        "primary_train": "historical_arm_env",
        "portfolio_arm": "eq",
        "om_touch_enabled": True,
        "hedge_leg_spread_bps": 5.0,
        "policy_mode": "archetype_crisis",
        "cao_c": 1.5,
        "kappa": 1.0,
        "train_env_steps": 16,
        "train_episodes": 1,
        "train_epochs": 1,
        "n_minibatches": 1,
        "weight_head": "softmax",
        "n_assets": K,
        "projection_mode": "soft",
        "claim_tier": "research",
    }
    rat.train_research_hist(returns, factors, cfg, seed=0)
    assert captured
    assert abs(captured[0] - 3.0) < 1e-9


def test_research_train_inflation_uses_term_spread_z(monkeypatch) -> None:
    import numpy as np

    import mascotrl.eval.research_alpha_train as rat
    from mascotrl.policy.objective_factory import episode_weights as real_ew

    captured: list[float] = []

    def spy_ew(mode, G, *, cao_c=1.5, kappa=1.0, **kwargs):
        captured.append(float(cao_c))
        return real_ew(mode, G, cao_c=cao_c, kappa=kappa, **kwargs)

    monkeypatch.setattr(rat, "episode_weights", spy_ew)

    T, K = 32, 4
    rng = np.random.default_rng(2)
    returns = rng.normal(0.0, 0.01, size=(T, K)).astype(np.float64)
    factors = rng.normal(0.0, 0.01, size=(T, 4)).astype(np.float64)
    cfg = {
        "algo": "ppo",
        "architecture": "mlp",
        "objective": "mean_std_cao",
        "objective_primary": True,
        "primary_train": "historical_arm_env",
        "portfolio_arm": "eq",
        "om_touch_enabled": True,
        "hedge_leg_spread_bps": 5.0,
        "policy_mode": "archetype_inflation",
        "term_spread_z": 2.0,
        "cao_c": 1.5,
        "kappa": 1.0,
        "train_env_steps": 16,
        "train_episodes": 1,
        "train_epochs": 1,
        "n_minibatches": 1,
        "weight_head": "softmax",
        "n_assets": K,
        "projection_mode": "soft",
        "claim_tier": "research",
    }
    rat.train_research_hist(returns, factors, cfg, seed=0)
    assert captured
    expected = apply_risk_aversion(1.5, "archetype_inflation", term_spread_z=2.0)
    assert abs(captured[0] - expected) < 1e-9


def test_risk_aversion_multiplier_inflation_clip() -> None:
    m = risk_aversion_multiplier("archetype_inflation", term_spread_z=10.0)
    assert abs(m - (1.0 + 0.25 * 2.0)) < 1e-9
