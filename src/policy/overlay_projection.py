"""Overlay CMDP projection with configurable delta modes for spectrum arms."""
from __future__ import annotations

import cvxpy as cp
import torch
import torch.nn as nn
from cvxpylayers.torch import CvxpyLayer

from src.policy.convex_projection import ConvexProjectionLayer

ALLOWED_DELTA_MODES = frozenset({"soft", "joint", "option_block", "off"})


class OverlayProjectionLayer(nn.Module):
    """Single-book projection with optional / joint / option-block delta modes.

    Modes:
      - soft: delegate to ``ConvexProjectionLayer`` (status-quo options arm).
      - joint: ``|w · Δ| ≤ s_δ`` with caller-supplied Δ (equity slots = 1).
      - option_block: zero equity Δ entries so only the option block is constrained.
      - off: drop the delta constraint; turnover and box remain.

    Joint turnover ``||w - w_prev||_1 ≤ τ + s_τ`` always spans all slots.
    """

    def __init__(
        self,
        num_assets: int,
        *,
        delta_mode: str = "soft",
        option_slots: int | None = None,
        turnover_limit: float = 0.15,
        penalty_weight: float = 1e4,
        vol_scale_floor: float = 0.05,
        vol_scale_cap: float = 8.0,
        lambda_eps: float = 1e-2,
        max_name_abs_weight: float = 5.0,
    ):
        super().__init__()
        if delta_mode not in ALLOWED_DELTA_MODES:
            raise ValueError(f"unknown delta_mode={delta_mode!r}")
        self.K = int(num_assets)
        self.delta_mode = str(delta_mode)
        self.option_slots = (
            self.K if option_slots is None else int(option_slots)
        )
        if self.delta_mode == "option_block" and not (0 < self.option_slots <= self.K):
            raise ValueError("option_block requires 0 < option_slots <= num_assets")
        self.tau0 = float(turnover_limit)
        self.tau = self.tau0
        self.penalty = float(penalty_weight)
        self.vol_scale_floor = float(vol_scale_floor)
        self.vol_scale_cap = float(vol_scale_cap)
        self.lambda_eps = float(lambda_eps)
        self.max_name = float(max_name_abs_weight)

        if self.delta_mode == "soft":
            self._soft = ConvexProjectionLayer(
                self.K,
                turnover_limit=self.tau0,
                penalty_weight=self.penalty,
                vol_scale_floor=self.vol_scale_floor,
                vol_scale_cap=self.vol_scale_cap,
                lambda_eps=self.lambda_eps,
                max_name_abs_weight=self.max_name,
            )
            self.cvx_layer = None
            self._off_layer = None
            return

        self._soft = None
        w_raw = cp.Parameter(self.K)
        w_prev = cp.Parameter(self.K)
        lambda_slack = cp.Parameter(nonneg=True)
        tau_limit = cp.Parameter(nonneg=True)
        w_name_max = cp.Parameter(nonneg=True)
        w_exec = cp.Variable(self.K)
        s_turnover = cp.Variable(nonneg=True)

        if self.delta_mode == "off":
            objective = cp.Minimize(
                cp.sum_squares(w_exec - w_raw) + lambda_slack * s_turnover
            )
            constraints = [
                cp.norm(w_exec - w_prev, 1) <= tau_limit + s_turnover,
                cp.abs(w_exec) <= w_name_max,
            ]
            problem = cp.Problem(objective, constraints)
            self._off_layer = CvxpyLayer(
                problem,
                parameters=[w_raw, w_prev, lambda_slack, tau_limit, w_name_max],
                variables=[w_exec, s_turnover],
            )
            self.cvx_layer = None
            self.register_buffer("_delta_mask", torch.ones(self.K))
            return

        self._off_layer = None
        deltas = cp.Parameter(self.K)
        s_delta = cp.Variable(nonneg=True)
        objective = cp.Minimize(
            cp.sum_squares(w_exec - w_raw)
            + lambda_slack * s_delta
            + lambda_slack * s_turnover
        )
        constraints = [
            cp.norm(w_exec - w_prev, 1) <= tau_limit + s_turnover,
            cp.abs(w_exec) <= w_name_max,
            cp.abs(w_exec @ deltas) <= s_delta,
        ]
        problem = cp.Problem(objective, constraints)
        self.cvx_layer = CvxpyLayer(
            problem,
            parameters=[w_raw, w_prev, deltas, lambda_slack, tau_limit, w_name_max],
            variables=[w_exec, s_delta, s_turnover],
        )

        mask = torch.ones(self.K)
        if self.delta_mode == "option_block":
            mask[self.option_slots :] = 0.0
        self.register_buffer("_delta_mask", mask)

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
        return self.penalty / (vol + self.lambda_eps)

    def _mask_deltas(self, deltas: torch.Tensor) -> torch.Tensor:
        if self.delta_mode == "option_block":
            return deltas * self._delta_mask.to(device=deltas.device, dtype=deltas.dtype)
        if self.delta_mode == "off":
            return torch.zeros_like(deltas)
        return deltas

    def _analytic_fallback(
        self,
        w_raw: torch.Tensor,
        w_prev: torch.Tensor,
        deltas: torch.Tensor,
        tau: float,
    ) -> torch.Tensor:
        if self.delta_mode == "off":
            proj = w_raw
        else:
            d = self._mask_deltas(deltas)
            d_norm2 = (d * d).sum(dim=-1, keepdim=True).clamp_min(1e-8)
            proj = w_raw - ((w_raw * d).sum(dim=-1, keepdim=True) / d_norm2) * d
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
        if self.delta_mode == "soft":
            return self._soft(
                w_raw,
                w_prev,
                deltas,
                vol_scale=vol_scale,
                turnover_limit=turnover_limit,
                return_slacks=return_slacks,
            )

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

        if self.delta_mode == "off":
            try:
                w_exec, s_turn = self._off_layer(w_raw, w_prev, lam, tau, w_cap)
                if return_slacks:
                    zero = torch.zeros(B, device=device, dtype=dtype)
                    return w_exec, zero, s_turn
                return w_exec
            except Exception:
                w_fb = self._analytic_fallback(w_raw, w_prev, deltas, float(tau.mean()))
                if return_slacks:
                    zero = torch.zeros(B, device=device, dtype=dtype)
                    return w_fb, zero, zero
                return w_fb

        d_in = self._mask_deltas(deltas)
        d_scale = d_in.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        d_hat = d_in / d_scale
        try:
            w_exec, s_delta, s_turn = self.cvx_layer(
                w_raw, w_prev, d_hat, lam, tau, w_cap
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
