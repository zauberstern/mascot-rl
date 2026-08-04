"""transition_source must change CMDPEnv physics (anti-stamp regression)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("cpp_rbergomi")


class _DummyFE(torch.nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, raw, iv_feat):
        B, K, _, D = raw.shape
        return torch.zeros(B, K, D)


class _DummyPolicy:
    pass


def _env_for_world(world: str, seed: int = 0):
    from src.env.cmdp_env import CMDPEnv
    from src.simulator import get_world_bundle

    cfg = {
        "n_paths": 2,
        "n_assets": 2,
        "n_steps": 16,
        "n_strikes": 3,
        "n_maturities": 2,
        "seed": seed,
        "train_world": world if world != "rbergomi" else "rbergomi",
        "force_world_bundle": True,
        "garch_n_inner": 128,
    }
    if world == "historical":
        # Synthetic OM-like panel via GBM spots stamped as historical.
        # Use a different seed so physics differ from the gbm cell.
        bundle = get_world_bundle({**cfg, "train_world": "gbm", "seed": seed + 99})
        surfaces = bundle["surfaces"]
        spots = bundle["spot_paths"]
        ivs = bundle["atm_iv_paths"]
        src = "historical"
    elif world == "rbergomi":
        bundle = get_world_bundle({**cfg, "train_world": "rbergomi", "force_world_bundle": True})
        surfaces = bundle["surfaces"]
        spots = bundle["spot_paths"]
        ivs = bundle["atm_iv_paths"]
        src = "rbergomi"
    else:
        bundle = get_world_bundle({**cfg, "train_world": world})
        surfaces = bundle["surfaces"]
        spots = bundle["spot_paths"]
        ivs = bundle["atm_iv_paths"]
        src = world

    d_model = 8
    fe = _DummyFE(d_model)
    env = CMDPEnv(
        surfaces,
        fe,
        _DummyPolicy(),
        d_model=d_model,
        macro_dim=4,
        use_gpu=False,
        transition_source=src,
        spot_paths=None if src == "rbergomi" else spots,
        atm_iv_paths=None if src == "rbergomi" else ivs,
        execution_spread_bps=0.0,
        execution_impact_coef=0.0,
    )
    return env


def _reward_traj(env, seed: int = 0) -> np.ndarray:
    torch.manual_seed(seed)
    env.reset(path=0, start_t=1, episode_seed=seed)
    rewards = []
    done = False
    while not done:
        w = torch.full((1, env.K), 1.0 / env.K)
        out = env.step(w)
        rewards.append(float(out.reward.item()))
        done = bool(out.done)
    return np.asarray(rewards, dtype=np.float64)


def test_transition_source_requires_spot_paths():
    """Declared gbm is allowed at construction; reset/step fail closed without paths."""
    from src.env.cmdp_env import CMDPEnv

    surfaces = torch.rand(1, 2, 8, 3, 2)
    env = CMDPEnv(
        surfaces,
        _DummyFE(4),
        _DummyPolicy(),
        d_model=4,
        macro_dim=2,
        use_gpu=False,
        transition_source="gbm",
        spot_paths=None,
    )
    assert env.transition_source == "gbm"
    with pytest.raises(ValueError, match="requires spot_paths"):
        env.reset(path=0, start_t=1)


def test_transition_source_physics_differ():
    worlds = ["rbergomi", "gbm", "heston", "garch", "sabr", "historical"]
    trajs = {}
    for w in worlds:
        env = _env_for_world(w, seed=42)
        trajs[w] = _reward_traj(env, seed=42)
    # Pairwise L1 distance must exceed 1e-6 for all distinct pairs.
    names = list(trajs)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            # Align lengths
            n = min(trajs[a].size, trajs[b].size)
            dist = float(np.abs(trajs[a][:n] - trajs[b][:n]).sum())
            assert dist > 1e-6, f"{a} vs {b} identical physics (L1={dist})"
