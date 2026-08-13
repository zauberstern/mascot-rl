"""Thin single-agent RL adapters (PPO / SAC / TD3 / DDPG / MCPG / RRL / DQN)
for the ``algo`` spectrum axis bakeoff.

Shared obs/action interface; smoke-trainable without lake data.
Architecture axis bodies live in ``src.policy.bodies`` so PPO and
off-policy adapters share one trunk builder (Part D.2).
"""
from __future__ import annotations

from typing import Any

from mascotrl.policy.single_agent_agents import (
    DDPGAgent,
    DQNAgent,
    MCPGAgent,
    PPOAgent,
    RRLAgent,
    SACAgent,
    TD3Agent,
    _REGISTRY,
)
from mascotrl.policy.single_agent_common import (
    ActorCritic as _ActorCritic,
    AssetTemporalActorCritic as _AssetTemporalActorCritic,
    BaseAgent as _BaseAgent,
    apply_weight_head as _apply_weight_head,
    compute_gae,
)

__all__ = [
    "DDPGAgent",
    "DQNAgent",
    "MCPGAgent",
    "PPOAgent",
    "RRLAgent",
    "SACAgent",
    "TD3Agent",
    "_apply_weight_head",
    "compute_gae",
    "make_single_agent",
]


def _registry_with_cppo() -> dict[str, type[_BaseAgent]]:
    out = dict(_REGISTRY)
    try:
        from mascotrl.policy.cppo import CPPOAgent

        out["cppo"] = CPPOAgent
    except ImportError:
        pass
    try:
        from mascotrl.policy.omnisafe_adapter import OmniSafeCPPOAgent

        out["cppo_omnisafe"] = OmniSafeCPPOAgent
    except ImportError:
        pass
    return out


def make_single_agent(
    algo: str,
    *,
    obs_dim: int,
    action_dim: int,
    lr: float = 3e-4,
    rl_backend: str | None = None,
    **kwargs: Any,
) -> _BaseAgent:
    key = str(algo).lower().strip()
    # Default matches resolve_rl_backend (sb3). Production train path passes an
    # explicit resolved backend; callers that omit rl_backend get SB3 when
    # supported. Set rl_backend="custom" to force hand-rolled PyTorch.
    raw_backend = rl_backend if rl_backend is not None else kwargs.pop("rl_backend", None)
    if raw_backend is None:
        from mascotrl.policy.sb3_adapter import resolve_rl_backend

        backend = resolve_rl_backend(None)
    else:
        backend = str(raw_backend).lower().strip()

    arch = str(kwargs.get("architecture", "mlp") or "mlp").lower().strip()
    temporal = arch in {"gru", "lstm", "mamba", "mamba2", "transformer"}

    if backend == "sb3":
        from mascotrl.logging_utils import get_logger

        _log = get_logger("mascotrl.policy.single_agent")
        # RecurrentPPO train_epoch is stubbed; fall back to custom temporal body.
        if key == "ppo" and arch in {"gru", "lstm"}:
            _log.warning(
                "SB3 RecurrentPPO train_epoch is not implemented; "
                "falling back to custom for architecture=%s",
                arch,
            )
        # SB3 MLP path: skip when a temporal body is required (mamba/transformer/
        # sac+gru etc. stay on the custom AssetTemporalPolicyBody path).
        elif key in frozenset({"ppo", "sac", "td3", "ddpg", "dqn", "ppo_recurrent"}) and (
            not temporal or key == "dqn"
        ):
            try:
                from mascotrl.policy.sb3_adapter import make_sb3_agent

                return make_sb3_agent(
                    key, obs_dim=obs_dim, action_dim=action_dim, lr=lr, **kwargs
                )
            except ImportError as exc:
                _log.warning(
                    "SB3 import failed for algo=%s; falling back to custom: %s", key, exc
                )
            except ValueError as exc:
                _log.warning(
                    "SB3 unavailable for algo=%s (%s); falling back to custom", key, exc
                )

    registry = _registry_with_cppo()
    try:
        cls = registry[key]
    except KeyError as exc:
        raise KeyError(f"unknown single-agent algo: {algo!r}") from exc
    agent = cls(obs_dim, action_dim, lr=lr, **kwargs)  # type: ignore[call-arg]
    # Provenance: custom agents always stamp backend=custom so train metadata
    # reflects the implementation that actually ran (not the cfg request).
    if not hasattr(agent, "backend"):
        agent.backend = "custom"  # type: ignore[attr-defined]
    return agent
