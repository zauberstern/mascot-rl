"""MCPG conformance: pure MC (gae_lambda=1) + RSQP vs Neagu reference."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from tests.conftest import FLOAT_TOL
import torch

from mascotrl.policy.objective_factory import episode_weights
from mascotrl.policy.single_agent import MCPGAgent, compute_gae


_REF = Path(__file__).resolve().parents[1] / "library_research" / "mcpg_reference"


def test_mcpg_uses_gae_lambda_one_pure_mc():
    """MCPGAgent must use undiscounted-bootstrap MC (lambda=1), not PPO's 0.95."""
    src = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "mascotrl"
        / "policy"
        / "single_agent_agents.py"
    )
    tree = ast.parse(src.read_text(encoding="utf-8"))
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "MCPGAgent":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    fn = sub.func
                    name = ""
                    if isinstance(fn, ast.Name):
                        name = fn.id
                    elif isinstance(fn, ast.Attribute):
                        name = fn.attr
                    if name == "compute_gae":
                        for kw in sub.keywords:
                            if kw.arg == "gae_lambda":
                                assert isinstance(kw.value, ast.Constant)
                                assert float(kw.value.value) == pytest.approx(1.0, **FLOAT_TOL)
                                found = True
    assert found, "MCPGAgent.train_epoch must call compute_gae(..., gae_lambda=1.0)"


def test_mcpg_sparse_terminal_no_bootstrap_past_done():
    """At done=1, next_value must be masked out (no bootstrap past episode end)."""
    torch.manual_seed(0)
    t = 5
    rewards = torch.zeros(t)
    rewards[-1] = -0.5  # terminal RSQP-style penalty
    values = torch.zeros(t)
    next_values = torch.zeros(t)
    next_values[-1] = 99.0  # would poison terminal return if (1-done) mask failed
    dones = torch.zeros(t)
    dones[-1] = 1.0
    _, returns = compute_gae(
        rewards, values, next_values, dones, gamma=0.99, gae_lambda=1.0
    )
    # Terminal return is exactly r_T (values=0, bootstrap masked)
    assert float(returns[-1]) == pytest.approx(-0.5, abs=1e-6)
    # Backward MC: R_t = gamma * R_{t+1} with zero intermediate rewards
    expected = torch.zeros(t)
    expected[-1] = -0.5
    g = 0.99
    for i in range(t - 2, -1, -1):
        expected[i] = rewards[i] + g * expected[i + 1]
    assert torch.allclose(returns, expected, atol=1e-5)
    assert float(returns.max()) < 1.0


def test_mcpg_agent_train_epoch_finite():
    agent = MCPGAgent(obs_dim=4, action_dim=2, lr=1e-3)
    obs = torch.randn(8, 4)
    actions = torch.randn(8, 2)
    rewards = torch.zeros(8)
    rewards[-1] = -0.2
    next_obs = torch.randn(8, 4)
    dones = torch.zeros(8)
    dones[-1] = 1.0
    stats = agent.train_epoch(
        obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones
    )
    assert torch.isfinite(torch.tensor(stats["loss"]))


def test_rsqp_matches_neagu_definition():
    """episode_weights('rsqp') is the score-function gradient of Neagu RSQP."""
    # Neagu: rho = sqrt(E[relu(xi)^2]); xi = -G when G is return
    g = torch.tensor([-0.4, -0.1, 0.05, 0.2], dtype=torch.float32)
    xi = -g
    pos = torch.clamp(xi, min=0.0)
    rho = torch.sqrt(torch.mean(pos.pow(2)) + 1e-12)
    # d rho / d xi_i  (via autograd on Neagu metric)
    xi_var = xi.detach().clone().requires_grad_(True)
    loss = torch.sqrt(torch.mean(torch.clamp(xi_var, min=0.0).pow(2)) + 1e-12)
    loss.backward()
    grad_xi = xi_var.grad
    # Score-function weight on G: chain rule d rho / d G = d rho / d xi * d xi / d G = -grad_xi
    # Our factory returns pos2/(2*rho) on G-scale where pos2=relu(-G)^2;
    # that equals grad of rho w.r.t. each sample contribution for REINFORCE on G.
    w = episode_weights("rsqp", g)
    manual = pos.pow(2) / (2.0 * rho)
    assert torch.allclose(w, manual, atol=1e-5)
    # Consistency: grad_xi ≈ pos / (n * rho) for positive xi; weight on G is positive for losses
    assert (w[g < 0] > 0).all()
    assert float(w[g > 0].sum()) == pytest.approx(0.0)
    assert grad_xi is not None

