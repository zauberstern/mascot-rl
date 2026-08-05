"""Part D.2: architecture body shared across PPO / SAC / TD3 / DDPG."""
from __future__ import annotations

import torch

from mascotrl.policy.bodies import (
    AssetTemporalPolicyBody,
    MLPPolicyBody,
    body_backend_name,
    build_policy_body,
)
from mascotrl.policy.single_agent import make_single_agent


def _obs_spec(architecture: str, *, k: int = 4, d_model: int = 8, seq: int = 2):
    if architecture == "mlp":
        return {"obs_dim": k * d_model, "action_dim": k, "hidden": 32}
    return {
        "obs_dim": k * seq * d_model,
        "action_dim": k,
        "num_assets": k,
        "d_model": d_model,
        "seq_len": seq,
        "hidden": 32,
        "with_critic": False,
    }


def test_build_policy_body_mlp_and_temporal_classes() -> None:
    mlp = build_policy_body("mlp", _obs_spec("mlp"), {})
    assert isinstance(mlp, MLPPolicyBody)
    assert body_backend_name(mlp) == "mlp"
    for arch in ("gru", "lstm", "transformer", "mamba"):
        body = build_policy_body(arch, _obs_spec(arch), {})
        assert isinstance(body, AssetTemporalPolicyBody)
        assert body_backend_name(body) == arch


def test_temporal_bodies_under_ppo_sac_td3_ddpg() -> None:
    k, d_model, seq = 3, 6, 2
    obs_dim = k * seq * d_model
    for arch in ("gru", "lstm", "transformer", "mamba"):
        for algo in ("ppo", "sac", "td3", "ddpg"):
            agent = make_single_agent(
                algo,
                obs_dim=obs_dim,
                action_dim=k,
                architecture=arch,
                num_assets=k,
                d_model=d_model,
                seq_len=seq,
                hidden=16,
                rl_backend="custom",
            )
            if algo == "ppo":
                assert getattr(agent, "architecture") == arch
                body = agent.net.body if hasattr(agent.net, "body") else agent.net
                assert isinstance(body, AssetTemporalPolicyBody)
                assert body.temporal_backend == arch
            else:
                assert getattr(agent, "architecture") == arch
                assert isinstance(agent.actor, (AssetTemporalPolicyBody, MLPPolicyBody))
                assert body_backend_name(agent.actor) == arch
            x = torch.randn(2, obs_dim)
            with torch.no_grad():
                a = agent.act(x, deterministic=True)
            assert a.shape[-1] == k


def test_temporal_bodies_under_mcpg_rrl() -> None:
    """Wave 5: MCPG/RRL consume arch_kwargs via shared body builder."""
    k, d_model, seq = 3, 6, 2
    obs_dim = k * seq * d_model
    for arch in ("gru", "lstm"):
        mcpg = make_single_agent(
            "mcpg",
            obs_dim=obs_dim,
            action_dim=k,
            architecture=arch,
            num_assets=k,
            d_model=d_model,
            seq_len=seq,
            hidden=16,
        )
        assert mcpg.architecture == arch
        body = mcpg.net.body if hasattr(mcpg.net, "body") else mcpg.net
        assert isinstance(body, AssetTemporalPolicyBody)
        rrl = make_single_agent(
            "rrl",
            obs_dim=obs_dim,
            action_dim=k,
            architecture=arch,
            num_assets=k,
            d_model=d_model,
            seq_len=seq,
            hidden=16,
        )
        assert rrl.architecture == arch
        assert isinstance(rrl.actor, AssetTemporalPolicyBody)
        x = torch.randn(2, obs_dim)
        with torch.no_grad():
            assert mcpg.act(x, deterministic=True).shape[-1] == k
            assert rrl.act(x, deterministic=True).shape[-1] == k


def test_dqn_requires_discrete_refuses_non_mlp() -> None:
    from mascotrl.spectrum.registry import get_option, validate_cfg
    import pytest

    opt = get_option("algo", "dqn")
    assert opt.requires_discrete is True
    with pytest.raises(ValueError, match="requires_discrete"):
        validate_cfg({"algo": "dqn", "architecture": "gru", "objective": "differential_sharpe"})
    assert validate_cfg({"algo": "dqn", "architecture": "mlp", "objective": "differential_sharpe"})[
        "algo"
    ] == "dqn"
