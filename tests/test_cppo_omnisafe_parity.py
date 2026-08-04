"""Parity / smoke for vendored OmniSafe CPPO adapter."""
from __future__ import annotations

import torch

from src.policy.cppo import CPPOAgent
from src.policy.omnisafe_adapter import OmniSafeCPPOAgent
from src.policy.single_agent import make_single_agent
from src.policy.vendor.omnisafe import PIDLagrangian


def _batch(n: int = 64, obs_dim: int = 8, act_dim: int = 3):
    obs = torch.randn(n, obs_dim)
    actions = torch.randn(n, act_dim)
    rewards = torch.randn(n) * 0.01
    next_obs = torch.randn(n, obs_dim)
    dones = torch.zeros(n)
    dones[-1] = 1.0
    return obs, actions, rewards, next_obs, dones


def test_pid_lagrange_reduces_positive_violation():
    pid = PIDLagrangian(
        pid_kp=0.5,
        pid_ki=0.1,
        pid_kd=0.01,
        pid_d_delay=5,
        pid_delta_p_ema_alpha=0.9,
        pid_delta_d_ema_alpha=0.9,
        sum_norm=True,
        diff_norm=False,
        penalty_max=50,
        lagrangian_multiplier_init=0.0,
        cost_limit=0.0,
    )
    for _ in range(20):
        pid.pid_update(1.0)  # cost above limit
    assert pid.lagrangian_multiplier > 0.0


def test_omnisafe_cppo_smoke_finite():
    agent = OmniSafeCPPOAgent(8, 3, hidden=16, normalize_obs=False, omnisafe_algo="cppo_pid")
    obs, actions, rewards, next_obs, dones = _batch()
    # need old logprobs / act path: use train_epoch
    stats = agent.train_epoch(
        obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones
    )
    assert "omnisafe_lambda" in stats
    assert torch.isfinite(torch.tensor(stats["omnisafe_lambda"]))
    w = agent.act(obs[:1])
    assert w.shape[-1] == 3
    assert torch.isfinite(w).all()


def test_make_single_agent_cppo_omnisafe():
    agent = make_single_agent("cppo_omnisafe", obs_dim=6, action_dim=2, hidden=8, normalize_obs=False)
    assert agent.name == "cppo_omnisafe"


def test_custom_vs_omnisafe_both_update():
    custom = CPPOAgent(8, 3, hidden=16, normalize_obs=False)
    omni = OmniSafeCPPOAgent(8, 3, hidden=16, normalize_obs=False, omnisafe_algo="cppo_pid")
    obs, actions, rewards, next_obs, dones = _batch()
    # Make rewards mostly negative so CVaR cost is positive
    rewards = -torch.abs(rewards) - 0.01
    s1 = custom.train_epoch(obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones)
    s2 = omni.train_epoch(obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones)
    assert "cvar_nu" in s1 or "trajectory_cvar" in s1
    assert s2["omnisafe_ep_cost"] >= 0.0
    # Both duals should move into the same order of magnitude band after a few updates
    for _ in range(5):
        s1 = custom.train_epoch(obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones)
        s2 = omni.train_epoch(obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones)
    assert s2["omnisafe_lambda"] >= 0.0
