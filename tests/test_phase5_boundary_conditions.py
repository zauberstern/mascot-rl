"""Phase 5 item 42: boundary-condition tests (learning_starts, act, ckpt, phase2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.plumbing
from tests.conftest import FLOAT_TOL
import torch

pytest.importorskip("stable_baselines3")
pytest.importorskip("gymnasium")

from mascotrl.eval.research_alpha_train import _maybe_resume_checkpoint, _save_checkpoint
from mascotrl.policy.single_agent import make_single_agent


def _policy_params(agent) -> list[torch.Tensor]:
    for attr in ("net", "actor", "q"):
        mod = getattr(agent, attr, None)
        if isinstance(mod, torch.nn.Module):
            return [p.detach().clone() for p in mod.parameters()]
    raise AssertionError(f"no policy module on {type(agent).__name__}")


def _perturb_policy(agent) -> None:
    with torch.no_grad():
        for attr in ("net", "actor", "q"):
            mod = getattr(agent, attr, None)
            if isinstance(mod, torch.nn.Module):
                for p in mod.parameters():
                    p.add_(0.25)
                return
    raise AssertionError(f"no policy module on {type(agent).__name__}")


# --- 1. SB3 off-policy learning_starts boundary ---


@pytest.mark.parametrize("algo", ["sac", "td3", "ddpg"])
@pytest.mark.parametrize(
    "t,expect_stub",
    [
        (31, True),   # learning_starts - 1
        (32, False),  # learning_starts
        (33, False),  # learning_starts + 1
    ],
)
def test_sb3_offpolicy_learning_starts_boundary(algo: str, t: int, expect_stub: bool):
    """Below starts may stub; at/above must hit real train without logger AttributeError."""
    agent = make_single_agent(algo, obs_dim=8, action_dim=2, lr=1e-3, rl_backend="sb3")
    assert int(agent._model.learning_starts) == 32
    stats = agent.train_epoch(
        obs=torch.randn(t, 8),
        actions=torch.randn(t, 2),
        rewards=torch.randn(t),
        next_obs=torch.randn(t, 8),
        dones=torch.zeros(t),
    )
    assert "optimizer_steps" in stats
    if expect_stub:
        assert stats.get("loss_source") == "stub"
        assert float(stats["optimizer_steps"]) == pytest.approx(0.0, **FLOAT_TOL)
    else:
        assert stats.get("loss_source") != "stub"
        assert float(stats["optimizer_steps"]) >= 1.0


# --- 2. act batch sizes + DQN zero weights ---


@pytest.mark.parametrize("algo,backend", [("ppo", "custom"), ("ppo", "sb3"), ("dqn", "custom")])
@pytest.mark.parametrize("batch", [1, 4])
def test_act_batch_size_1_and_gt1(algo: str, backend: str, batch: int):
    agent = make_single_agent(algo, obs_dim=6, action_dim=2, lr=1e-3, rl_backend=backend)
    w = agent.act(torch.randn(batch, 6), deterministic=True)
    assert w.shape[0] == batch
    assert torch.isfinite(w).all()


def test_dqn_raw_to_weights_all_zero():
    from mascotrl.policy.single_agent import DQNAgent

    agent = DQNAgent(obs_dim=4, action_dim=3, lr=1e-3)
    w = agent.raw_to_weights(torch.zeros(2, 3))
    np.testing.assert_allclose(w.detach().numpy(), 0.0, atol=1e-12)


# --- 3. Checkpoint save/resume roundtrip ---


@pytest.mark.parametrize(
    "algo,backend",
    [
        ("ppo", "custom"),
        ("ppo", "sb3"),
        ("sac", "custom"),
        ("td3", "custom"),
        ("ddpg", "custom"),
        ("dqn", "custom"),
        ("mcpg", "custom"),
        ("rrl", "custom"),
        ("cppo", "custom"),
    ],
)
def test_checkpoint_roundtrip_phase5(algo: str, backend: str, tmp_path: Path):
    kw: dict = {"obs_dim": 6, "action_dim": 2, "lr": 1e-3, "rl_backend": backend}
    if algo in ("ppo", "sac", "td3", "ddpg", "mcpg", "rrl", "cppo", "dqn"):
        kw["hidden"] = 8
    if algo == "cppo":
        kw["normalize_obs"] = False
    src = make_single_agent(algo, **kw)
    _perturb_policy(src)
    before = _policy_params(src)

    cfg = {
        "_checkpoint_dir": str(tmp_path),
        "_fold_id": 0,
        "_run_config_hash": f"p5-{algo}-{backend}",
    }
    _save_checkpoint(src, cfg, seed=0, episode=1, optimizer_steps=1)
    ckpts = list(tmp_path.glob("*.pt"))
    assert ckpts, f"no checkpoint written for {algo}/{backend}"

    dst = make_single_agent(algo, **kw)
    _maybe_resume_checkpoint(
        dst,
        {"_resume_checkpoint": str(ckpts[0]), "_run_config_hash": cfg["_run_config_hash"]},
    )
    after = _policy_params(dst)
    assert len(before) == len(after)
    for a, b in zip(before, after):
        assert torch.allclose(a, b), f"{algo}/{backend} param mismatch after resume"


# --- 4. Phase 2 coverage cross-check (resume skip / hybrid no-overwrite / OOS) ---


def test_phase2_resume_episode_skip_still_registered():
    import tests.test_phase2_eval_hardening as p2

    assert callable(p2.test_intra_fold_resume_skips_completed_episodes)


def test_phase2_hybrid_no_overwrite_still_registered():
    """Warm-started agent must skip resume (hybrid pretrain+finetune no-overwrite)."""
    import tests.test_phase2_eval_hardening as p2

    assert callable(p2.test_warm_started_agent_skips_resume)


def test_phase2_oos_residualizer_still_registered():
    import tests.test_phase2_eval_hardening as p2

    assert callable(p2.test_oos_residualizer_requires_train_frozen)
    assert callable(p2.test_oos_uses_train_residualizer_betas)
