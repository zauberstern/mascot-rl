"""Model bundle export and HAPPO replay."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.registry import ModelCard, save_model_bundle, zoo_root
from src.policy.single_agent import make_single_agent


def test_happo_act_weights_no_not_implemented(tmp_path, monkeypatch):
    monkeypatch.setenv("MASCOTRL_MODEL_ZOO", str(tmp_path))
    from src.policy.happo import HAPPOEngine
    from src.models.inference import act_weights, HAPPO_OOS_REPLAY_SUPPORTED

    assert HAPPO_OOS_REPLAY_SUPPORTED is True
    k, d_model, macro = 3, 8, 4
    engine = HAPPOEngine(k, enriched_dim=d_model, macro_dim=macro)
    card = ModelCard(
        model_id="happo-ppo-eq-0-deadbee",
        family="happo",
        algo="happo",
        obs_dim=k * d_model,
        action_dim=k,
        n_assets=k,
        seed=0,
    )
    deploy = {"n_assets": k, "d_model": d_model, "macro_dim": macro}
    save_model_bundle({"policy": engine.state_dict()}, card, root=tmp_path, deploy_config=deploy)
    obs = np.zeros(k * d_model, dtype=np.float32)
    w = act_weights(card.model_id, obs, root=tmp_path)
    assert w.shape == (k,)
    assert np.isfinite(w).all()


def test_export_onnx_single_agent(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    monkeypatch.setenv("MASCOTRL_MODEL_ZOO", str(tmp_path))
    from src.models.export import export_onnx

    agent = make_single_agent(
        "ppo", obs_dim=10, action_dim=2, rl_backend="custom", hidden=16, normalize_obs=False
    )
    card = ModelCard(
        model_id="research_single_agent-ppo-eq-0-abc12345",
        family="research_single_agent",
        algo="ppo",
        obs_dim=10,
        action_dim=2,
        n_assets=2,
        seed=0,
    )
    net = agent.net
    save_model_bundle(
        {"policy": net.state_dict()},
        card,
        root=tmp_path,
        obs_schema={"obs_dim": 10, "action_dim": 2},
        deploy_config={"algo": "ppo", "weight_head": "softmax", "rl_backend": "custom", "ppo_hidden": 16},
    )
    onnx_path = export_onnx(card.model_id, root=tmp_path)
    assert onnx_path.is_file()


def test_ship_model_writes_complete_dir(tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    monkeypatch.setenv("MASCOTRL_MODEL_ZOO", str(tmp_path))
    from scripts.ship_model import ship_model
    from src.models.export import export_onnx

    agent = make_single_agent(
        "ppo", obs_dim=6, action_dim=2, rl_backend="custom", hidden=8, normalize_obs=False
    )
    card = ModelCard(
        model_id="research_single_agent-ppo-eq-1-def67890",
        family="research_single_agent",
        algo="ppo",
        obs_dim=6,
        action_dim=2,
        seed=1,
    )
    save_model_bundle(
        {"policy": agent.net.state_dict()},
        card,
        root=tmp_path,
        obs_schema={"obs_dim": 6},
        deploy_config={"rl_backend": "custom", "ppo_hidden": 8},
    )
    export_onnx(card.model_id, root=tmp_path)
    out = ship_model(card.model_id, out_dir=tmp_path / "ship", root=tmp_path)
    for name in ("weights.pt", "card.json", "policy.onnx", "predict.py", "ship_manifest.json"):
        assert (out / name).is_file()
