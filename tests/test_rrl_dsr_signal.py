"""RRL agent signal equals standalone DifferentialSharpe; no double-DSR."""
from __future__ import annotations

import pytest
import torch

from src.eval.differential_sharpe import DifferentialSharpe
from src.eval.yaml_honesty import refuse_rrl_double_dsr
from src.policy.single_agent import RRLAgent


def test_rrl_signal_matches_standalone_dsr():
    torch.manual_seed(0)
    rewards = torch.tensor([0.01, -0.02, 0.015, 0.0, 0.03], dtype=torch.float32)
    dones = torch.zeros_like(rewards)
    eta = 0.02
    ds = DifferentialSharpe(eta=eta)
    expected = torch.zeros_like(rewards)
    for i, r in enumerate(rewards.tolist()):
        expected[i] = ds.step(r)

    agent = RRLAgent(obs_dim=3, action_dim=2, eta=eta, lr=1e-3)
    ds2 = DifferentialSharpe(eta=eta)
    signal = torch.zeros_like(rewards)
    for i in range(rewards.shape[0]):
        signal[i] = ds2.step(float(rewards[i].item()))
    assert torch.allclose(signal, expected, atol=1e-12)

    obs = torch.randn(rewards.shape[0], 3)
    actions = torch.randn(rewards.shape[0], 2)
    stats = agent.train_epoch(
        obs=obs,
        actions=actions,
        rewards=rewards,
        next_obs=torch.randn_like(obs),
        dones=dones,
    )
    assert "mean_diff_sharpe" in stats


def test_rrl_episode_boundary_resets_dsr():
    eta = 0.05
    rewards = torch.tensor([0.02, -0.01, 0.03, -0.02], dtype=torch.float32)
    dones = torch.tensor([0.0, 1.0, 0.0, 1.0])
    ds = DifferentialSharpe(eta=eta)
    expected = []
    for i, r in enumerate(rewards.tolist()):
        if i > 0 and dones[i - 1] > 0.5:
            ds = DifferentialSharpe(eta=eta)
        expected.append(ds.step(r))

    ds_cont = DifferentialSharpe(eta=eta)
    cont = [ds_cont.step(float(r)) for r in rewards.tolist()]
    assert cont[2] != expected[2]


def test_refuse_rrl_double_dsr():
    with pytest.raises(ValueError, match="double-DSR"):
        refuse_rrl_double_dsr({"algo": "rrl", "objective": "differential_sharpe"})
    refuse_rrl_double_dsr({"algo": "rrl", "objective": "mtm_pnl"})
    refuse_rrl_double_dsr({"algo": "ppo", "objective": "differential_sharpe"})
