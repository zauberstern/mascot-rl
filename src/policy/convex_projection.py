"""Differentiable CMDP projection with vol-scaled slack variables."""
from __future__ import annotations

import cvxpy as cp
import torch
import torch.nn as nn
from cvxpylayers.torch import CvxpyLayer


# @lat: [[core#CMDP projection]]
class ConvexProjectionLayer(nn.Module):
    """
    Elastic / slack-augmented CMDP projection (Slater-safe KKT gradients).

        min ||w - w_raw||² + λ(σ) (s_δ + s_τ)
        s.t. |w·Δ̂| ≤ s_δ
             ||w - w_prev||₁ ≤ τ + s_τ
             |w_i| ≤ w_name_max   (hard box; aligned with ExternalRiskGuard)
             s ≥ 0

    Elastic law: non-negative slacks permanently expand the feasible cone so
    the QP never aborts under rough-vol shocks (undefined KKT → NaN grads).
    Large λ drives s→0 whenever the hard constraints are attainable.

    Dynamic law (expert): λ(σ) = λ₀ / (σ + ε)
    Higher instantaneous local vol → *lower* slack penalty → more room to
    satisfy Slater without gradient blow-ups, while calm regimes keep
    constraints tight.

    ``tau`` is a CVXPY Parameter so turnover limits can be episode-adaptive
    without rebuilding the DPP graph (Elastic OdynLayer-style flexibility).
    """

    def __init__(
        self,
        num_assets: int,
        turnover_limit: float = 0.15,
        penalty_weight: float = 1e4,
        vol_scale_floor: float = 0.05,
        vol_scale_cap: float = 8.0,
        lambda_eps: float = 1e-2,
        max_name_abs_weight: float = 5.0,
        scs_max_iters: int = 250,
    ):
        super().__init__()
        self.K = num_assets
        self.tau0 = float(turnover_limit)
        self.tau = self.tau0  # backward-compatible alias
        self.penalty = float(penalty_weight)
        self.vol_scale_floor = float(vol_scale_floor)
        self.vol_scale_cap = float(vol_scale_cap)
        self.lambda_eps = float(lambda_eps)
        self.max_name = float(max_name_abs_weight)
        self.scs_max_iters = int(scs_max_iters)

        w_raw = cp.Parameter(self.K)
        w_prev = cp.Parameter(self.K)
        deltas = cp.Parameter(self.K)
        # λ(σ) passed in already inverted — keeps the CVXPY graph DPP-clean.
        lambda_slack = cp.Parameter(nonneg=True)
        tau_limit = cp.Parameter(nonneg=True)
        w_name_max = cp.Parameter(nonneg=True)

        w_exec = cp.Variable(self.K)
        s_delta = cp.Variable(nonneg=True)
        s_turnover = cp.Variable(nonneg=True)

        objective = cp.Minimize(
            cp.sum_squares(w_exec - w_raw)
            + lambda_slack * s_delta
            + lambda_slack * s_turnover
        )
        constraints = [
            cp.abs(w_exec @ deltas) <= s_delta,
            cp.norm(w_exec - w_prev, 1) <= tau_limit + s_turnover,
            cp.abs(w_exec) <= w_name_max,
        ]
        problem = cp.Problem(objective, constraints)
        self.cvx_layer = CvxpyLayer(
            problem,
            parameters=[w_raw, w_prev, deltas, lambda_slack, tau_limit, w_name_max],
            variables=[w_exec, s_delta, s_turnover],
        )

    def _resolve_vol(
        self, deltas: torch.Tensor, vol_scale: torch.Tensor | float | None
    ) -> torch.Tensor:
        B = deltas.shape[0]
        device, dtype = deltas.device, deltas.dtype
        if vol_scale is None:
            vs = deltas.detach().abs().mean(dim=-1)
        elif isinstance(vol_scale, (float, int)):
            vs = torch.full((B,), float(vol_scale), device=device, dtype=dtype)
        else:
            vs = vol_scale.detach().reshape(-1).to(device=device, dtype=dtype)
            if vs.numel() == 1 and B > 1:
                vs = vs.expand(B)
            elif vs.numel() != B:
                vs = vs.mean().expand(B)
        return vs.clamp(self.vol_scale_floor, self.vol_scale_cap)

    def _lambda_from_vol(self, vol: torch.Tensor) -> torch.Tensor:
        # λ = λ₀ / (σ + ε) — high vol loosens slack pressure.
        return self.penalty / (vol + self.lambda_eps)

    def _analytic_fallback(
        self,
        w_raw: torch.Tensor,
        w_prev: torch.Tensor,
        deltas: torch.Tensor,
        tau: float,
    ) -> torch.Tensor:
        d_norm2 = (deltas * deltas).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        proj = w_raw - ((w_raw * deltas).sum(dim=-1, keepdim=True) / d_norm2) * deltas
        diff = proj - w_prev
        l1 = diff.abs().sum(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = torch.clamp(tau / l1, max=1.0)
        out = w_prev + diff * scale
        return out.clamp(-self.max_name, self.max_name)

    def forward(
        self,
        w_raw: torch.Tensor,
        w_prev: torch.Tensor,
        deltas: torch.Tensor,
        vol_scale: torch.Tensor | float | None = None,
        turnover_limit: torch.Tensor | float | None = None,
        return_slacks: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d_scale = deltas.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        d_hat = deltas / d_scale
        vs = self._resolve_vol(deltas, vol_scale)
        lam = self._lambda_from_vol(vs)
        B = deltas.shape[0]
        device, dtype = deltas.device, deltas.dtype
        if turnover_limit is None:
            tau = torch.full((B,), self.tau0, device=device, dtype=dtype)
        elif isinstance(turnover_limit, (float, int)):
            tau = torch.full((B,), float(turnover_limit), device=device, dtype=dtype)
        else:
            tau = turnover_limit.detach().reshape(-1).to(device=device, dtype=dtype)
            if tau.numel() == 1 and B > 1:
                tau = tau.expand(B)
        w_cap = torch.full((B,), self.max_name, device=device, dtype=dtype)
        try:
            w_exec, s_delta, s_turn = self.cvx_layer(
                w_raw,
                w_prev,
                d_hat,
                lam,
                tau,
                w_cap,
                solver_args={
                    "solve_method": "SCS",
                    "max_iters": int(self.scs_max_iters),
                    "eps": 1e-3,
                    "verbose": False,
                },
            )
            if return_slacks:
                return w_exec, s_delta, s_turn
            return w_exec
        except Exception:
            w_fb = self._analytic_fallback(w_raw, w_prev, deltas, float(tau.mean()))
            if return_slacks:
                zero = torch.zeros(B, device=device, dtype=dtype)
                return w_fb, zero, zero
            return w_fb
