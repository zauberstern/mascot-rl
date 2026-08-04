"""Phase 3 systemic hardening: resume hash, registry, provenance."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.plumbing


def test_spectrum_resume_hash_includes_weight_head_and_lr():
    from scripts.run_spectrum_campaign import _spectrum_run_config_hash

    base = {
        "algo": "ppo",
        "architecture": "mlp",
        "objective": "mean_std_cao",
        "reward": "residual_pnl",
        "n_assets": 100,
        "weight_head": "softmax",
        "lr": 3e-4,
        "gamma": 0.99,
        "policy_mode": "soft",
        "universe_arm": "eq",
    }
    h1 = _spectrum_run_config_hash(base)
    h2 = _spectrum_run_config_hash({**base, "weight_head": "tanh_l1"})
    h3 = _spectrum_run_config_hash({**base, "lr": 1e-3})
    h4 = _spectrum_run_config_hash({**base, "gamma": 0.95})
    assert h1 != h2, "weight_head change must invalidate resume hash"
    assert h1 != h3, "lr change must invalidate resume hash"
    assert h1 != h4, "gamma change must invalidate resume hash"


def test_spectrum_resume_hash_includes_container_digest(monkeypatch):
    from scripts.run_spectrum_campaign import (
        _SPECTRUM_RESUME_HASH_KEYS,
        _spectrum_run_config_hash,
    )

    assert "container_digest" in _SPECTRUM_RESUME_HASH_KEYS
    base = {"algo": "ppo", "n_assets": 10}
    monkeypatch.delenv("MASCOTRL_CONTAINER_DIGEST", raising=False)
    h0 = _spectrum_run_config_hash(base)
    monkeypatch.setenv("MASCOTRL_CONTAINER_DIGEST", "sha256:abc")
    h1 = _spectrum_run_config_hash(base)
    h2 = _spectrum_run_config_hash({**base, "container_digest": "sha256:xyz"})
    assert h0 != h1
    assert h1 != h2


def test_validate_cfg_refuses_cppo_with_mtm_pnl():
    from src.spectrum.registry import validate_cfg

    with pytest.raises(ValueError, match="requires_episode_returns|cppo"):
        validate_cfg({"algo": "cppo", "objective": "mtm_pnl", "architecture": "mlp"})


def test_validate_cfg_refuses_rrl_differential_sharpe_reward():
    from src.spectrum.registry import validate_cfg

    with pytest.raises(ValueError, match="rrl|differential_sharpe"):
        validate_cfg(
            {
                "algo": "rrl",
                "objective": "mtm_pnl",
                "reward": "differential_sharpe",
                "architecture": "mlp",
            }
        )


def test_validate_cfg_refuses_illegal_weight_head_for_algo():
    from src.spectrum.registry import ALGO_HEADS, validate_cfg

    assert "dirichlet_entropy" not in ALGO_HEADS["ppo"]
    with pytest.raises(ValueError, match="weight_head|illegal"):
        validate_cfg(
            {
                "algo": "ppo",
                "objective": "mean_std_cao",
                "architecture": "mlp",
                "weight_head": "dirichlet_entropy",
            }
        )


def test_validate_cfg_allows_cherrypick_ppo_dirichlet_mean():
    """Cherrypick Sweep C exception: ppo + dirichlet_mean is legal."""
    from src.spectrum.registry import validate_cfg

    out = validate_cfg(
        {
            "algo": "ppo",
            "objective": "mean_std_cao",
            "architecture": "mlp",
            "weight_head": "dirichlet_mean",
            "head_axis_id": "dirichlet_mean",
        }
    )
    assert out["algo"] == "ppo"


def test_validate_cfg_refuses_happo_screening_without_stamp():
    from src.spectrum.registry import validate_cfg

    with pytest.raises(ValueError, match="happo_screening_requires_dispatch_stamp"):
        validate_cfg(
            {
                "algo": "happo",
                "objective": "mean_std_cao",
                "architecture": "mlp",
                "claim_tier": "research",
                "protocol_tier": "screening",
                "train_env_steps": 25000,
            }
        )


def test_validate_cfg_accepts_happo_screening_with_stamp():
    from src.spectrum.registry import validate_cfg

    out = validate_cfg(
        {
            "algo": "happo",
            "objective": "mean_std_cao",
            "architecture": "mlp",
            "claim_tier": "research",
            "protocol_tier": "screening",
            "train_env_steps": 25000,
            "happo_dispatch_only": True,
        }
    )
    assert out["algo"] == "happo"


def test_generator_stamps_happo_dispatch_only():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "generate_spectrum_grid.py"
    spec = importlib.util.spec_from_file_location("spectrum_grid_gen", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod._cell_cfg(
        arm="eq",
        k=100,
        algo="happo",
        body="mlp",
        head="delta_w",
        objective="mean_std_cao",
        tier="screening",
    )
    assert cfg.get("happo_dispatch_only") is True


def test_validate_cell_cfg_rejects_broken_generated_cell():
    """Item 28: deliberately broken cell raises under validate_cell_cfg."""
    import importlib.util
    from pathlib import Path

    from src.spectrum.cell_schema import validate_cell_cfg

    path = Path(__file__).resolve().parents[1] / "scripts" / "generate_spectrum_grid.py"
    spec = importlib.util.spec_from_file_location("spectrum_grid_gen", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod._cell_cfg(
        arm="eq",
        k=100,
        algo="ppo",
        body="mlp",
        head="softmax",
        objective="mean_std_cao",
        tier="screening",
    )
    cfg["bogus_phase3_key"] = 1
    with pytest.raises(ValueError, match="unknown keys"):
        validate_cell_cfg(cfg)


def test_load_policy_fail_closed_without_rl_backend(tmp_path):
    """Item 29: deploy_config.json without rl_backend raises."""
    import torch

    from src.models.inference import load_policy
    from src.models.registry import ModelCard, make_model_id, save_model_bundle
    from src.policy.single_agent import make_single_agent
    from src.eval.research_alpha_train import _agent_policy_module

    agent = make_single_agent(
        "ppo", obs_dim=4, action_dim=2, rl_backend="custom", hidden=8, normalize_obs=False
    )
    mid = make_model_id(
        family="research_single_agent",
        algo="ppo",
        arm="eq",
        seed=0,
        run_config_hash="p3infer01",
    )
    card = ModelCard(
        model_id=mid,
        family="research_single_agent",
        algo="ppo",
        arm="eq",
        obs_dim=4,
        action_dim=2,
        n_assets=2,
        seed=0,
        run_config_hash="p3infer01",
    )
    net = _agent_policy_module(agent)
    save_model_bundle(
        {"policy": net.state_dict()},
        card,
        root=tmp_path,
        deploy_config={"algo": "ppo", "ppo_hidden": 8},  # missing rl_backend
    )
    with pytest.raises(ValueError, match="rl_backend"):
        load_policy(mid, root=tmp_path)


def test_train_research_hist_stamps_actual_rl_backend_for_dqn():
    """DQN always trains custom; artifact must not lie with cfg default sb3."""
    import numpy as np

    from src.eval.research_alpha_train import train_research_hist

    rng = np.random.default_rng(0)
    t, k = 24, 2
    rets = rng.normal(0.0002, 0.01, size=(t, k))
    fac = rng.normal(0.0, 0.01, size=(t, 4))
    cfg = {
        "primary_train": "historical_arm_env",
        "portfolio_arm": "eq",
        "n_assets": k,
        "train_epochs": 1,
        "policy": "single_agent",
        "projection_mode": "soft",
        "algo": "dqn",
        "objective": "mtm_pnl",
        "lr": 1e-3,
        # omit rl_backend -> defaults to sb3 request, but agent is custom
    }
    out = train_research_hist(rets, fac, cfg, seed=0)
    assert out["rl_backend"] == "custom"
    assert getattr(out["agent"], "backend", None) == "custom"
