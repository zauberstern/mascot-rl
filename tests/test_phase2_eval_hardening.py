"""Phase 2 systemic bug-hunt: eval pipeline fail-closed hardening (items 7-22)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.plumbing
import torch


def _toy_panel(t: int = 40, k: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, 0.01, size=(t, k))
    factors = rng.normal(0.0, 0.01, size=(t, 4))
    return rets, factors


def _base_cfg(**extra):
    cfg = {
        "primary_train": "historical_arm_env",
        "portfolio_arm": "eq",
        "n_assets": 3,
        "projection_mode": "soft",
        "rl_backend": "custom",
        "algo": "ppo",
        "objective": "mtm_pnl",
        "train_episodes": 1,
        "train_epochs": 1,
        "n_minibatches": 1,
        "ppo_hidden": 8,
        "equity_bps": 0.0,
    }
    cfg.update(extra)
    return cfg


# --- Item 7 ---


def test_gym_env_step_passes_through_zero_weights():
    from mascotrl.eval.research_alpha_train import build_research_hist_env
    from mascotrl.policy.sb3_adapter import GymnasiumHistoricalEnv

    rets, fac = _toy_panel(t=20, k=3)
    inner = build_research_hist_env(rets, fac, _base_cfg())
    env = GymnasiumHistoricalEnv(inner)
    env.reset(seed=0)
    steps: list[np.ndarray] = []
    orig = inner.step

    def capture(w):
        steps.append(np.asarray(w, dtype=np.float64).copy())
        return orig(w)

    inner.step = capture  # type: ignore[method-assign]
    env.step(np.zeros(3, dtype=np.float64))
    assert len(steps) == 1
    np.testing.assert_allclose(steps[0], 0.0, atol=1e-12)


def test_dqn_all_zero_levels_yield_zero_portfolio_weights():
    from mascotrl.policy.single_agent import DQNAgent

    agent = DQNAgent(obs_dim=4, action_dim=3, lr=1e-3)
    w = agent.raw_to_weights(torch.zeros(1, 3))
    np.testing.assert_allclose(w.detach().numpy(), 0.0, atol=1e-12)


def test_train_collect_passes_zero_weights_to_env(monkeypatch):
    from mascotrl.eval.research_alpha_train import build_research_hist_env, train_research_hist
    from mascotrl.policy.single_agent import DQNAgent

    rets, fac = _toy_panel(t=16, k=3)
    steps: list[np.ndarray] = []

    class ZeroDQN(DQNAgent):
        def act_and_logp_raw(self, obs, *, deterministic=False):
            b = obs.shape[0]
            return torch.zeros(b, self.action_dim), torch.zeros(b)

    env0 = build_research_hist_env(rets, fac, _base_cfg(algo="dqn"))
    obs_dim = int(np.asarray(env0.reset(seed=0)[0]).reshape(-1).size)
    agent = ZeroDQN(obs_dim=obs_dim, action_dim=3)

    import mascotrl.eval.research_alpha_train as rat

    real_build = rat.build_research_hist_env

    def build_capture(*a, **kw):
        env = real_build(*a, **kw)
        orig = env.step

        def step(w):
            steps.append(np.asarray(w, dtype=np.float64).copy())
            return orig(w)

        env.step = step  # type: ignore[method-assign]
        return env

    monkeypatch.setattr(rat, "build_research_hist_env", build_capture)
    train_research_hist(rets, fac, _base_cfg(algo="dqn"), seed=0, agent=agent)
    assert steps, "expected env.step calls"
    for w in steps:
        np.testing.assert_allclose(w, 0.0, atol=1e-12)


# --- Item 8 ---


def test_agent_policy_module_prefers_actor_over_q():
    from mascotrl.eval.research_alpha_train import _agent_policy_module
    from mascotrl.models.inference import _agent_policy_module as inf_mod
    from mascotrl.policy.single_agent import DDPGAgent

    agent = DDPGAgent(obs_dim=6, action_dim=3, hidden=8)
    assert _agent_policy_module(agent) is agent.actor
    assert inf_mod(agent) is agent.actor


def test_ddpg_checkpoint_roundtrip_preserves_actor(tmp_path: Path):
    from mascotrl.eval.research_alpha_train import _maybe_resume_checkpoint, _save_checkpoint
    from mascotrl.policy.single_agent import DDPGAgent

    src = DDPGAgent(obs_dim=6, action_dim=3, hidden=8)
    with torch.no_grad():
        for p in src.actor.parameters():
            p.add_(0.5)
    _save_checkpoint(
        src,
        {"_checkpoint_dir": str(tmp_path), "_fold_id": 0, "_run_config_hash": "h"},
        seed=0,
        episode=1,
        optimizer_steps=1,
    )
    dst = DDPGAgent(obs_dim=6, action_dim=3, hidden=8)
    _maybe_resume_checkpoint(
        dst,
        {
            "_resume_checkpoint": str(next(tmp_path.glob("*.pt"))),
            "_run_config_hash": "h",
        },
    )
    for a, b in zip(src.actor.parameters(), dst.actor.parameters()):
        assert torch.allclose(a, b)


def test_sac_checkpoint_roundtrip_preserves_critic(tmp_path: Path):
    from mascotrl.eval.research_alpha_train import _maybe_resume_checkpoint, _save_checkpoint
    from mascotrl.policy.single_agent import SACAgent

    src = SACAgent(obs_dim=6, action_dim=3, hidden=8)
    with torch.no_grad():
        for p in src.q1.parameters():
            p.add_(0.25)
        src.log_alpha.add_(0.1)
    _save_checkpoint(
        src,
        {"_checkpoint_dir": str(tmp_path), "_fold_id": 1, "_run_config_hash": "sac"},
        seed=1,
        episode=2,
        optimizer_steps=3,
    )
    dst = SACAgent(obs_dim=6, action_dim=3, hidden=8)
    _maybe_resume_checkpoint(
        dst,
        {
            "_resume_checkpoint": str(next(tmp_path.glob("*.pt"))),
            "_run_config_hash": "sac",
        },
    )
    for a, b in zip(src.q1.parameters(), dst.q1.parameters()):
        assert torch.allclose(a, b)
    assert torch.allclose(src.log_alpha, dst.log_alpha)


def test_rrl_checkpoint_roundtrip_preserves_log_std(tmp_path: Path):
    from mascotrl.eval.research_alpha_train import _maybe_resume_checkpoint, _save_checkpoint
    from mascotrl.policy.single_agent import RRLAgent

    src = RRLAgent(obs_dim=6, action_dim=3, hidden=8)
    with torch.no_grad():
        src.log_std.fill_(-0.3)
    _save_checkpoint(
        src,
        {"_checkpoint_dir": str(tmp_path), "_fold_id": 2, "_run_config_hash": "rrl"},
        seed=2,
        episode=1,
        optimizer_steps=1,
    )
    dst = RRLAgent(obs_dim=6, action_dim=3, hidden=8)
    _maybe_resume_checkpoint(
        dst,
        {
            "_resume_checkpoint": str(next(tmp_path.glob("*.pt"))),
            "_run_config_hash": "rrl",
        },
    )
    assert torch.allclose(src.log_std, dst.log_std)


def test_td3_checkpoint_state_roundtrip():
    from mascotrl.policy.single_agent import TD3Agent

    src = TD3Agent(obs_dim=6, action_dim=3, hidden=8)
    with torch.no_grad():
        for p in src.actor_t.parameters():
            p.add_(0.1)
    blob = src.checkpoint_state()
    dst = TD3Agent(obs_dim=6, action_dim=3, hidden=8)
    dst.load_checkpoint_state(blob)
    for a, b in zip(src.actor_t.parameters(), dst.actor_t.parameters()):
        assert torch.allclose(a, b)


# --- Item 9 ---


def test_intra_fold_resume_skips_completed_episodes(monkeypatch, tmp_path: Path):
    from mascotrl.eval.research_alpha_train import (
        _save_checkpoint,
        build_research_hist_env,
        train_research_hist,
    )
    from mascotrl.policy.single_agent import make_single_agent

    rets, fac = _toy_panel(t=20, k=3)
    env = build_research_hist_env(rets, fac, _base_cfg())
    obs_dim = int(np.asarray(env.reset(seed=0)[0]).reshape(-1).size)
    agent = make_single_agent(
        "ppo", obs_dim=obs_dim, action_dim=3, hidden=8, rl_backend="custom"
    )
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    _save_checkpoint(
        agent,
        {
            "_checkpoint_dir": str(ckpt_dir),
            "_fold_id": 0,
            "_run_config_hash": "epskip",
        },
        seed=0,
        episode=2,
        optimizer_steps=1,
    )
    resets: list[int] = []
    import mascotrl.eval.research_alpha_train as rat

    real_build = rat.build_research_hist_env

    def build_track(*a, **kw):
        e = real_build(*a, **kw)
        orig = e.reset
        n = {"i": 0}

        def reset(*, seed=None):
            n["i"] += 1
            resets.append(n["i"])
            return orig(seed=seed)

        e.reset = reset  # type: ignore[method-assign]
        return e

    monkeypatch.setattr(rat, "build_research_hist_env", build_track)
    cfg = _base_cfg(
        train_episodes=3,
        _checkpoint_dir=str(ckpt_dir),
        _fold_id=0,
        _run_config_hash="epskip",
        _resume_checkpoint=str(next(ckpt_dir.glob("*.pt"))),
    )
    train_research_hist(rets, fac, cfg, seed=0)
    # probe reset (obs_dim) + one collect episode after skip of 0,1
    assert len(resets) <= 3


# --- Item 10 ---


def test_warm_started_agent_skips_resume(monkeypatch, tmp_path: Path):
    from mascotrl.eval.research_alpha_train import build_research_hist_env, train_research_hist
    from mascotrl.policy.single_agent import make_single_agent

    rets, fac = _toy_panel(t=16, k=3)
    env = build_research_hist_env(rets, fac, _base_cfg())
    obs_dim = int(np.asarray(env.reset(seed=0)[0]).reshape(-1).size)
    warm = make_single_agent(
        "ppo", obs_dim=obs_dim, action_dim=3, hidden=8, rl_backend="custom"
    )
    called = {"n": 0}
    import mascotrl.eval.research_alpha_train as rat

    def spy(agent, cfg):
        called["n"] += 1
        return None

    monkeypatch.setattr(rat, "_maybe_resume_checkpoint", spy)
    ckpt = tmp_path / "dummy.pt"
    torch.save({"policy": {}, "run_config_hash": "x", "episode": 1}, ckpt)
    train_research_hist(
        rets,
        fac,
        _base_cfg(_resume_checkpoint=str(ckpt), _run_config_hash="x"),
        seed=0,
        agent=warm,
    )
    assert called["n"] == 0


# --- Item 11 ---


def test_refuse_rrl_double_dsr_checks_reward_key():
    from mascotrl.eval.yaml_honesty import refuse_rrl_double_dsr

    with pytest.raises(ValueError, match="double-DSR"):
        refuse_rrl_double_dsr({"algo": "rrl", "reward": "differential_sharpe"})


def test_validate_cfg_refuses_rrl_differential_sharpe_reward():
    from mascotrl.spectrum.registry import validate_cfg

    with pytest.raises(ValueError, match="double-DSR|rrl"):
        validate_cfg(
            {"algo": "rrl", "reward": "differential_sharpe", "objective": "mtm_pnl"}
        )


def test_dsr_reward_plus_episode_weight_raises():
    from mascotrl.eval.research_alpha_train import train_research_hist

    with pytest.raises(ValueError, match="differential_sharpe|episode_weight|stack"):
        train_research_hist(
            *_toy_panel(t=16, k=3),
            _base_cfg(
                reward="differential_sharpe",
                objective="mean_std_cao",
                objective_primary=True,
            ),
            seed=0,
        )


# --- Item 12 ---


def test_unknown_policy_raises():
    from mascotrl.eval.research_alpha_train import train_research_hist

    with pytest.raises(ValueError, match="unknown policy"):
        train_research_hist(
            *_toy_panel(t=12, k=3),
            _base_cfg(policy="not_a_real_policy"),
            seed=0,
        )


def test_resolve_reward_mode_propagates_non_import_errors(monkeypatch):
    from mascotrl.eval.research_alpha_train import _resolve_reward_mode
    import mascotrl.policy.objective_factory as of

    monkeypatch.setattr(
        of,
        "resolve_objective_mode",
        lambda cfg, default="none": (_ for _ in ()).throw(
            ValueError("malformed objective")
        ),
    )
    with pytest.raises(ValueError, match="malformed"):
        _resolve_reward_mode({"objective": "mean_std_cao"})


# --- Item 17 ---


def test_prune_fold_checkpoints_scopes_to_fold_seed(tmp_path: Path):
    from mascotrl.eval.research_alpha_train import prune_fold_checkpoints
    import time

    for name in (
        "fold0_seed0_ep00001.pt",
        "fold0_seed0_ep00002.pt",
        "fold1_seed0_ep00001.pt",
        "fold1_seed0_ep00002.pt",
    ):
        p = tmp_path / name
        p.write_bytes(b"x")
        time.sleep(0.02)
        p.touch()

    deleted = prune_fold_checkpoints(tmp_path, keep_latest=1, fold_id=0, seed=0)
    assert deleted == 1
    assert (tmp_path / "fold0_seed0_ep00002.pt").exists()
    assert (tmp_path / "fold1_seed0_ep00001.pt").exists()
    assert (tmp_path / "fold1_seed0_ep00002.pt").exists()


# --- Item 18 ---


def test_misaligned_feature_extras_raise_at_env_build():
    from mascotrl.eval.research_alpha_train import build_research_hist_env

    with pytest.raises(ValueError, match="feature_extras|dollar_volume|mismatch"):
        build_research_hist_env(
            *_toy_panel(t=20, k=3),
            _base_cfg(
                use_equity_feature_cube=True,
                feature_extras={"dollar_volume": np.ones((10, 3))},
            ),
        )


def test_slice_feature_extras_raises_on_short_array():
    from mascotrl.eval.research_alpha_cpcv import _slice_feature_extras

    with pytest.raises(ValueError, match="feature_extras|iv|mismatch"):
        _slice_feature_extras(
            {"feature_extras": {"iv": np.ones((5, 3))}},
            np.array([0, 1, 2, 8]),
        )


# --- Item 19 ---


def test_oos_residualizer_requires_train_frozen():
    from mascotrl.eval.friction import FrictionSpec
    from mascotrl.models.inference import roll_oos_with_agent

    rets, fac = _toy_panel(t=30, k=3)
    dates = list(pd.bdate_range("2020-01-01", periods=30))

    class DummyAgent:
        def act(self, obs, *, deterministic=True):
            return torch.full((1, 3), 1.0 / 3)

    with pytest.raises(ValueError, match="residualizer|train_residualizer|train-frozen"):
        roll_oos_with_agent(
            returns=rets,
            factors=fac,
            dates=dates,
            idx=np.arange(10, 20),
            agent=DummyAgent(),
            cfg=_base_cfg(),
            friction=FrictionSpec(equity_bps=0.0),
        )


def test_oos_uses_train_residualizer_betas():
    from mascotrl.eval.friction import FrictionSpec
    from mascotrl.eval.research_alpha_train import build_research_hist_env
    from mascotrl.eval.residualization import fit_ff4_residualizer, freeze_residualizer
    from mascotrl.models.inference import roll_oos_with_agent

    rets, fac = _toy_panel(t=80, k=3, seed=1)
    dates = list(pd.bdate_range("2020-01-01", periods=80))
    train_idx = np.arange(0, 50)
    test_idx = np.arange(50, 65)
    train_resid = freeze_residualizer(
        fit_ff4_residualizer(
            np.nanmean(rets[train_idx], axis=1),
            fac[train_idx],
            fold_id="train_fold",
        ),
        "train_fold",
    )
    oos_leak = freeze_residualizer(
        fit_ff4_residualizer(
            np.nanmean(rets[test_idx], axis=1),
            fac[test_idx],
            fold_id="oos_leak",
        ),
        "oos_leak",
    )
    assert not np.allclose(train_resid.betas, oos_leak.betas)

    class DummyAgent:
        def act(self, obs, *, deterministic=True):
            return torch.full((1, 3), 1.0 / 3)

    captured = {}
    real_build = build_research_hist_env

    def capture(returns, factors, cfg, **kw):
        env = real_build(returns, factors, cfg, **kw)
        captured["fold_id"] = env.residualizer.fold_id
        captured["betas"] = np.asarray(env.residualizer.betas).copy()
        return env

    with patch("mascotrl.eval.research_alpha_train.build_research_hist_env", capture):
        roll_oos_with_agent(
            returns=rets,
            factors=fac,
            dates=dates,
            idx=test_idx,
            agent=DummyAgent(),
            cfg=_base_cfg(),
            friction=FrictionSpec(equity_bps=0.0),
            train_residualizer=train_resid,
        )
    assert captured.get("fold_id") == "train_fold"
    np.testing.assert_allclose(captured["betas"], train_resid.betas)


# --- Item 20 ---


def test_rebalance_cadence_without_mask_raises():
    from mascotrl.eval.research_alpha_train import build_research_hist_env

    with pytest.raises(ValueError, match="rebalance"):
        build_research_hist_env(
            *_toy_panel(t=20, k=3),
            _base_cfg(rebalance_cadence="monthly"),
        )


def test_rebalance_cadence_with_mask_ok():
    from mascotrl.eval.research_alpha_train import build_research_hist_env

    mask = np.zeros(20, dtype=bool)
    mask[::5] = True
    env = build_research_hist_env(
        *_toy_panel(t=20, k=3),
        _base_cfg(rebalance_cadence="monthly", _rebalance_mask=mask),
    )
    assert env.rebalance_mask is not None


# --- Item 21 ---


def test_resume_without_run_config_hash_raises(tmp_path: Path):
    from mascotrl.eval.research_alpha_train import _maybe_resume_checkpoint
    from mascotrl.policy.single_agent import make_single_agent

    agent = make_single_agent(
        "ppo", obs_dim=4, action_dim=2, hidden=4, rl_backend="custom"
    )
    ckpt = tmp_path / "c.pt"
    torch.save({"policy": agent.net.state_dict(), "run_config_hash": "h"}, ckpt)
    with pytest.raises((RuntimeError, ValueError), match="run_config_hash"):
        _maybe_resume_checkpoint(agent, {"_resume_checkpoint": str(ckpt)})


# --- Item 22 ---


def test_explicit_zero_mikkila_xi_honored(monkeypatch):
    from mascotrl.eval.research_alpha_train import train_research_hist
    import mascotrl.policy.objective_factory as of

    seen = {}

    def fake_mikkila(r, xi=1.0):
        seen["xi"] = float(xi)
        return r

    monkeypatch.setattr(of, "mikkila_asym_reward", fake_mikkila)
    import mascotrl.eval.research_alpha_train as rat

    monkeypatch.setattr(rat, "mikkila_asym_reward", fake_mikkila)
    train_research_hist(
        *_toy_panel(t=16, k=3),
        _base_cfg(objective="mikkila_asym", objective_primary=True, mikkila_xi=0.0),
        seed=0,
    )
    assert seen.get("xi") == 0.0, f"explicit 0.0 rewritten; seen={seen}"
