"""Model zoo registry + inference tests (Part B)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from mascotrl.models.inference import HAPPO_OOS_REPLAY_SUPPORTED, act_weights, load_policy
from mascotrl.models.registry import (
    ModelCard,
    list_models,
    make_model_id,
    save_model_bundle,
    verify_bundle,
    write_model_zoo_index,
)
from mascotrl.policy.single_agent import make_single_agent
from mascotrl.spectrum.registry import allowed_ids


def _payload_for(agent) -> dict:
    from mascotrl.eval.research_alpha_train import _agent_policy_module

    net = _agent_policy_module(agent)
    assert net is not None, f"no policy module on {type(agent)}"
    return {
        "policy": net.state_dict(),
        "optimizer": None,
        "seed": 0,
        "fold_id": 0,
        "run_config_hash": "testhash01",
        "episode": 1,
        "optimizer_steps": 1,
    }


def test_save_load_roundtrip_preserves_weights(tmp_path: Path):
    agent = make_single_agent("ppo", obs_dim=8, action_dim=4, rl_backend="custom")
    mid = make_model_id(
        family="research_single_agent",
        algo="ppo",
        arm="eq",
        seed=0,
        run_config_hash="testhash01",
    )
    card = ModelCard(
        model_id=mid,
        family="research_single_agent",
        algo="ppo",
        arm="eq",
        obs_dim=8,
        action_dim=4,
        n_assets=4,
        seed=0,
        run_config_hash="testhash01",
    )
    save_model_bundle(_payload_for(agent), card, root=tmp_path)
    loaded, card2 = load_policy(mid, root=tmp_path)
    assert card2.model_id == mid
    from mascotrl.eval.research_alpha_train import _agent_policy_module

    net0 = _agent_policy_module(agent).state_dict()
    net1 = _agent_policy_module(loaded).state_dict()
    for k in net0:
        assert torch.allclose(net0[k], net1[k])


def test_tampered_weights_fail_verify(tmp_path: Path):
    agent = make_single_agent("ppo", obs_dim=4, action_dim=3, rl_backend="custom")
    mid = make_model_id(
        family="research_single_agent",
        algo="ppo",
        arm="eq",
        seed=1,
        run_config_hash="abc12345xx",
    )
    card = ModelCard(
        model_id=mid,
        family="research_single_agent",
        algo="ppo",
        obs_dim=4,
        action_dim=3,
        seed=1,
        run_config_hash="abc12345xx",
    )
    d = save_model_bundle(_payload_for(agent), card, root=tmp_path)
    # Tamper
    (d / "weights.pt").write_bytes(b"not-a-real-checkpoint")
    try:
        verify_bundle(mid, root=tmp_path)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "sha256" in str(e).lower()


def test_act_weights_matches_direct_agent(tmp_path: Path):
    agent = make_single_agent("ppo", obs_dim=6, action_dim=3, rl_backend="custom")
    mid = make_model_id(
        family="research_single_agent",
        algo="ppo",
        arm="eq",
        seed=2,
        run_config_hash="deadbeef01",
    )
    card = ModelCard(
        model_id=mid,
        family="research_single_agent",
        algo="ppo",
        obs_dim=6,
        action_dim=3,
        seed=2,
        run_config_hash="deadbeef01",
    )
    save_model_bundle(_payload_for(agent), card, root=tmp_path)
    obs = np.linspace(-1, 1, 6, dtype=np.float32)
    w_bundle = act_weights(mid, obs, root=tmp_path)
    with torch.no_grad():
        direct = agent.act(torch.as_tensor(obs).unsqueeze(0), deterministic=True)
    w_direct = direct.detach().cpu().numpy().reshape(-1)
    w_direct = w_direct / max(float(np.sum(np.abs(w_direct))), 1e-12)
    assert np.allclose(w_bundle, w_direct, atol=1e-5)
    assert abs(float(np.sum(np.abs(w_bundle))) - 1.0) < 1e-5


def test_all_registry_algos_can_roundtrip_bundle(tmp_path: Path):
    for algo in allowed_ids("algo"):
        if algo == "happo":
            from mascotrl.policy.happo import HAPPOEngine
            from mascotrl.models.inference import HAPPOInferenceAgent

            k, d_model, macro = 3, 8, 4
            engine = HAPPOEngine(k, enriched_dim=d_model, macro_dim=macro)
            mid = make_model_id(
                family="happo",
                algo="happo",
                arm="eq",
                seed=0,
                run_config_hash=f"h{algo}01",
            )
            card = ModelCard(
                model_id=mid,
                family="happo",
                algo="happo",
                obs_dim=k * d_model,
                action_dim=k,
                n_assets=k,
                seed=0,
                run_config_hash=f"h{algo}01",
            )
            deploy = {"n_assets": k, "d_model": d_model, "macro_dim": macro}
            save_model_bundle(
                {"policy": engine.state_dict()},
                card,
                root=tmp_path,
                deploy_config=deploy,
            )
            loaded, card2 = load_policy(mid, root=tmp_path)
            assert isinstance(loaded, HAPPOInferenceAgent)
            assert card2.family == "happo"
            continue
        agent = make_single_agent(algo, obs_dim=5, action_dim=3, rl_backend="custom")
        mid = make_model_id(
            family="research_single_agent",
            algo=algo,
            arm="eq",
            seed=0,
            run_config_hash=f"r{algo}01",
        )
        card = ModelCard(
            model_id=mid,
            family="research_single_agent",
            algo=algo,
            obs_dim=5,
            action_dim=3,
            seed=0,
            run_config_hash=f"r{algo}01",
        )
        save_model_bundle(_payload_for(agent), card, root=tmp_path)
        loaded, _ = load_policy(mid, root=tmp_path)
        assert loaded is not None


def test_list_and_index(tmp_path: Path):
    agent = make_single_agent("ppo", obs_dim=4, action_dim=2, rl_backend="custom")
    mid = make_model_id(
        family="research_single_agent",
        algo="ppo",
        arm="eq",
        seed=9,
        run_config_hash="index001",
    )
    card = ModelCard(
        model_id=mid,
        family="research_single_agent",
        algo="ppo",
        obs_dim=4,
        action_dim=2,
        seed=9,
        run_config_hash="index001",
        sharpe_mean=0.5,
    )
    save_model_bundle(_payload_for(agent), card, root=tmp_path)
    cards = list_models(root=tmp_path, algo="ppo")
    assert any(c.model_id == mid for c in cards)
    idx = write_model_zoo_index(root=tmp_path)
    assert idx.is_file()
    assert mid in idx.read_text()


def test_happo_oos_replay_supported(tmp_path: Path):
    """HAPPO zoo bundles rebuild via deploy_config for act_weights."""
    assert HAPPO_OOS_REPLAY_SUPPORTED is True
    from mascotrl.policy.happo import HAPPOEngine

    k, d_model, macro = 2, 8, 4
    engine = HAPPOEngine(k, enriched_dim=d_model, macro_dim=macro)
    mid = make_model_id(
        family="happo",
        algo="happo",
        arm="eq",
        seed=0,
        run_config_hash="happoexcl01",
    )
    card = ModelCard(
        model_id=mid,
        family="happo",
        algo="happo",
        obs_dim=k * d_model,
        action_dim=k,
        n_assets=k,
        seed=0,
        run_config_hash="happoexcl01",
    )
    deploy = {"n_assets": k, "d_model": d_model, "macro_dim": macro}
    save_model_bundle(
        {"policy": engine.state_dict()}, card, root=tmp_path, deploy_config=deploy
    )
    w = act_weights(mid, np.zeros(k * d_model, dtype=np.float32), root=tmp_path)
    assert w.shape == (k,)
