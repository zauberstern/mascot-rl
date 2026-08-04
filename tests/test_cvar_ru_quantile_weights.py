"""cvar_ru episode weights use VaR tail set, not zeta=0."""
from __future__ import annotations

import pytest
from tests.conftest import FLOAT_TOL

import math

import torch

from src.policy.objective_factory import episode_weights


def test_cvar_ru_uses_tail_not_zero():
    g = torch.tensor([0.1, 0.2, -0.05, -0.3, 0.0, 0.15, -0.1, 0.05])
    alpha = 0.75
    w = episode_weights("cvar_ru", g, alpha=alpha)
    n = g.numel()
    k = max(1, int(math.ceil((1.0 - alpha) * n)))
    assert int((w > 0).sum()) == k
    # Worst losses (most negative G) receive the weight.
    worst = torch.topk(-g, k=k).indices
    assert torch.all(w[worst] > 0)
    # zeta=0 on all-positive G → all-zero weights (the old bug).
    g_pos = torch.tensor([0.1, 0.2, 0.3, 0.4])
    w_pos = episode_weights("cvar_ru", g_pos, alpha=0.75)
    assert float(w_pos.sum()) > 0.0
    w_zero = (1.0 / 0.25) * torch.relu(-g_pos - 0.0)
    assert float(w_zero.sum()) == pytest.approx(0.0, **FLOAT_TOL)
