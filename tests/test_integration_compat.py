"""Integration audit: library coexistence and I/O contracts (AWS CPU stack)."""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
import torch


def test_core_libraries_import_together():
    mods = [
        "purgedcv",
        "stable_baselines3",
        "sb3_contrib",
        "gymnasium",
        "torch",
        "numpy",
        "pandas",
        "sklearn",
    ]
    for name in mods:
        importlib.import_module(name if name != "sklearn" else "sklearn")
    # Vendored OmniSafe duals (no pip omnisafe / Safety-Gymnasium)
    from src.policy.vendor.omnisafe.lagrange import Lagrange
    from src.policy.vendor.omnisafe.pid_lagrange import PIDLagrangian

    assert Lagrange is not None and PIDLagrangian is not None
    # HARL optional but expected in research envs
    harl = pytest.importorskip("harl")
    assert harl is not None


def test_purgedcv_io_contract():
    from src.eval.cpcv import CPCVConfig
    from src.eval.cpcv_lib import build_cpcv_folds_lib

    n = 40
    dates = list(pd.bdate_range("2020-01-01", periods=n))
    folds = build_cpcv_folds_lib(
        dates,
        CPCVConfig(n_splits=5, n_test_groups=2, purge_days=2, embargo_days=1),
    )
    assert len(folds) >= 1
    for fold in folds:
        assert fold.n_train_days > 0 and fold.n_test_days > 0
        assert fold.train_windows and fold.test_windows
        # Windows are date ranges; train/test days must not exceed panel
        assert fold.n_train_days + fold.n_test_days <= n + fold.n_purged_days + fold.n_embargoed_days


def test_sb3_flatten_reshape_io_contract():
    from src.policy.sb3_adapter import PortfolioFeaturesExtractor

    k, seq, c = 2, 3, 4
    flat_dim = k * seq * c
    ext = PortfolioFeaturesExtractor(
        features_dim=16, num_assets=k, seq_len=seq, n_channels=c
    )
    import gymnasium as gym

    space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32)
    module = ext.cls(space)
    x = torch.randn(5, flat_dim)
    y = module(x)
    assert y.shape == (5, 16)


def test_recurrent_ppo_extractor_shape():
    pytest.importorskip("sb3_contrib")
    from src.policy.sb3_adapter import make_sb3_agent

    agent = make_sb3_agent(
        "ppo_recurrent", obs_dim=12, action_dim=2, num_assets=2, seq_len=2
    )
    w = agent.act(torch.randn(1, 12))
    assert w.shape[0] == 1
    assert torch.isfinite(w).all()


def test_harl_io_contract():
    pytest.importorskip("harl")
    from src.policy.harl_adapter import HistoricalArmHARLEnv, default_happo_args

    class _StubEnv:
        K = 2
        obs_dim = 8

        def __init__(self):
            self._t = 0

        def reset(self):
            self._t = 0
            return np.zeros(8, dtype=np.float32), {}

        def step(self, action):
            self._t += 1
            done = self._t >= 3
            return (
                np.zeros(8, dtype=np.float32),
                0.0,
                done,
                False,
                {},
            )

    args = default_happo_args(obs_dim=8, action_dim=1, n_agents=2)
    assert "hidden_sizes" in args
    wrapped = HistoricalArmHARLEnv(_StubEnv(), n_agents=2)
    assert wrapped.n_agents == 2
    assert len(wrapped.observation_space) == 2
    assert len(wrapped.action_space) == 2


def test_omnisafe_vendor_io_contract():
    from src.policy.omnisafe_adapter import CostShaper, OmniSafeCPPOAgent

    shaper = CostShaper(cvar_alpha=0.95)
    rewards = torch.tensor([0.1, -0.2, 0.05, -0.5])
    dones = torch.zeros_like(rewards)
    dones[-1] = 1.0
    cost = shaper.episode_cost(rewards, dones)
    assert cost >= 0.0
    agent = OmniSafeCPPOAgent(obs_dim=4, action_dim=2)
    w = agent.act(torch.randn(1, 4))
    assert torch.isfinite(w).all()


def test_feature_cube_routing_shapes():
    """(K, seq, C) flattens to obs_dim for SB3 / OmniSafe; per-agent for HARL."""
    k, seq, c = 4, 2, 3
    cube = torch.randn(1, k, seq, c)
    flat = cube.reshape(1, -1)
    assert flat.shape[-1] == k * seq * c
    per_agent = cube.reshape(1, k, seq * c)
    assert per_agent.shape == (1, k, seq * c)


def test_friction_to_objective_no_sign_flip():
    from src.arms import ArmSpec
    from src.eval.friction import apply_costs
    from src.policy.objective_factory import episode_weights

    w = torch.tensor([[0.6, 0.4]])
    out = apply_costs(
        w,
        torch.zeros(1, 2),
        torch.tensor([0.02, 0.01]),
        arm=ArmSpec(id="eq", option_slots=0, equity_slots=2, delta_mode="off"),
        equity_bps=10.0,
        om_touch_enabled=False,
    )
    assert out.net < out.gross  # costs deduct, no soft-fee collapse
    g = torch.tensor([out.net, out.net * 0.5, -0.01], dtype=torch.float32)
    ww = episode_weights("mean_std_cao", g)
    assert torch.isfinite(ww).all()


def test_arch_and_cpcv_backend_resolvers():
    from src.eval.arch_bootstrap import resolve_bootstrap_backend
    from src.eval.cpcv_backend import resolve_use_purgedcv

    assert resolve_bootstrap_backend({}) == "custom"
    assert resolve_use_purgedcv({"use_purgedcv": False}) is False


def test_omnisafe_pip_not_required():
    """Full omnisafe package must not be required (pandas/gymnasium pin conflict)."""
    import importlib.util

    assert importlib.util.find_spec("src.policy.vendor.omnisafe.lagrange") is not None
