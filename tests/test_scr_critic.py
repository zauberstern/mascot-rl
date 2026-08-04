"""WP-P4: SCR critic mix off/nocf/full with leak assertion."""
from __future__ import annotations

import pytest
from tests.conftest import FLOAT_TOL
import torch

from src.eval.scr_critic import (
    assert_psi_leak_safe,
    build_scr_returns,
    mix_critic_targets,
    resolve_scr_mix,
)
from src.policy.rasp_locks import assert_rasp_locks


def _batch(n: int = 8) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    rewards = torch.randn(n)
    values = torch.randn(n)
    next_values = torch.randn(n)
    dones = torch.zeros(n)
    dones[-1] = 1.0
    return rewards, values, next_values, dones


def test_resolve_scr_full_refuses_non_ppo() -> None:
    with pytest.raises(ValueError, match="scr_full_requires_ppo_historical"):
        resolve_scr_mix({"scr_mix": "full", "algo": "sac", "train_world": "historical"})


def test_resolve_scr_full_refuses_non_historical() -> None:
    with pytest.raises(ValueError, match="scr_full_requires_ppo_historical"):
        resolve_scr_mix({"scr_mix": "full", "algo": "ppo", "train_world": "rbergomi"})


def test_leak_assertion_on_indices() -> None:
    assert_psi_leak_safe([0, 1, 3], t=3)
    with pytest.raises(ValueError, match="scr_psi_lookahead"):
        assert_psi_leak_safe([0, 1, 5], t=3)


def test_off_nocf_full_targets() -> None:
    rewards, values, next_values, dones = _batch()
    adv0, ret0, m0 = build_scr_returns(
        rewards=rewards,
        values=values,
        next_values=next_values,
        dones=dones,
        scr_mix="off",
    )
    adv_n, ret_n, m_n = build_scr_returns(
        rewards=rewards,
        values=values,
        next_values=next_values,
        dones=dones,
        scr_mix="nocf",
    )
    # nocf with identity equals off returns
    assert torch.allclose(ret0, ret_n, atol=1e-6)
    assert m0["actor_reward_checksum"] == m_n["actor_reward_checksum"]

    y_cf = ret0 + 1.0
    adv_f, ret_f, m_f = build_scr_returns(
        rewards=rewards,
        values=values,
        next_values=next_values,
        dones=dones,
        scr_mix="full",
        scr_beta=0.5,
        y_cf=y_cf,
        psi_indices_by_t=lambda t: list(range(t + 1)),
    )
    assert not torch.allclose(ret_f, ret0, atol=1e-4)
    expected = mix_critic_targets(ret0, y_cf, beta=0.5)
    assert torch.allclose(ret_f, expected, atol=1e-5)
    assert m_f["scr_beta"] == pytest.approx(0.5, **FLOAT_TOL)
    # Actor reward checksum identical across modes (reward never mixed).
    assert m0["actor_reward_checksum"] == m_f["actor_reward_checksum"]


def test_rasp_locks_scr_full_mcpg() -> None:
    with pytest.raises(ValueError, match="scr_full_requires_ppo_historical"):
        assert_rasp_locks(
            {
                "algo": "mcpg",
                "train_world": "historical",
                "scr_mix": "full",
                "projection_mode": "hard",
                "turnover_limit": 0.15,
            }
        )
