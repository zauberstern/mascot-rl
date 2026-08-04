# Vendored from OmniSafe (Apache-2.0).
# Upstream: https://github.com/PKU-Alignment/omnisafe
# Commit: 15603dd7a654a991d0a4648216b69d60b81a6366
# Source: omnisafe/common/lagrange.py
# Stripped of Safety-Gymnasium / omnisafe package imports.
"""Naive Lagrangian multiplier (PPOLag dual)."""
from __future__ import annotations

import torch


class Lagrange:
    """Lagrangian multiplier update for constrained policy gradients.

    ``update_lagrange_multiplier(Jc)`` steps
    ``λ ← λ + η · (Jc - cost_limit)`` with projection onto ``[0, upper]``.
    """

    def __init__(
        self,
        cost_limit: float,
        lagrangian_multiplier_init: float,
        lambda_lr: float,
        lambda_optimizer: str = "Adam",
        lagrangian_upper_bound: float | None = None,
    ) -> None:
        self.cost_limit: float = float(cost_limit)
        self.lambda_lr: float = float(lambda_lr)
        self.lagrangian_upper_bound: float | None = (
            float(lagrangian_upper_bound) if lagrangian_upper_bound is not None else None
        )
        init_value = max(float(lagrangian_multiplier_init), 0.0)
        self.lagrangian_multiplier = torch.nn.Parameter(
            torch.as_tensor(init_value),
            requires_grad=True,
        )
        self.lambda_range_projection = torch.nn.ReLU()
        if not hasattr(torch.optim, lambda_optimizer):
            raise AssertionError(f"Optimizer={lambda_optimizer} not found in torch.")
        torch_opt = getattr(torch.optim, lambda_optimizer)
        self.lambda_optimizer = torch_opt([self.lagrangian_multiplier], lr=lambda_lr)

    def compute_lambda_loss(self, mean_ep_cost: float) -> torch.Tensor:
        return -self.lagrangian_multiplier * (mean_ep_cost - self.cost_limit)

    def update_lagrange_multiplier(self, Jc: float) -> None:
        self.lambda_optimizer.zero_grad()
        lambda_loss = self.compute_lambda_loss(Jc)
        lambda_loss.backward()
        self.lambda_optimizer.step()
        hi = self.lagrangian_upper_bound
        if hi is None:
            self.lagrangian_multiplier.data.clamp_(0.0, float("inf"))
        else:
            self.lagrangian_multiplier.data.clamp_(0.0, hi)
