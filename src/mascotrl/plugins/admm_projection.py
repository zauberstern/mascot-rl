"""ADMM / Dykstra elastic CMDP projection (opt-in; cvxpy remains status quo)."""
from __future__ import annotations

import torch
import torch.nn as nn


def _project_box(w: torch.Tensor, cap: float) -> torch.Tensor:
    return w.clamp(-cap, cap)


def _project_l1_ball(v: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    """
    Euclidean projection onto {x : ||x||_1 <= r} (Duchi et al. ICML 2008).

    v: (B, K), radius: (B,) or scalar broadcast.
    """
    B, K = v.shape
    r = radius.reshape(-1).to(device=v.device, dtype=v.dtype)
    if r.numel() == 1:
        r = r.expand(B)
    abs_v = v.abs()
    l1 = abs_v.sum(dim=-1)
    out = v.clone()
    need = l1 > r + 1e-12
    if not need.any():
        return out
    # Sort descending for rows that need projection.
    for b in torch.where(need)[0].tolist():
        u = abs_v[b]
        ru = r[b]
        if float(u.sum()) <= float(ru) + 1e-12:
            continue
        us, _ = torch.sort(u, descending=True)
        cssv = torch.cumsum(us, dim=0) - ru
        ind = torch.arange(1, K + 1, device=v.device, dtype=v.dtype)
        cond = us - cssv / ind > 0
        if not cond.any():
            out[b] = 0.0
            continue
        rho = int(cond.nonzero()[-1].item())
        theta = cssv[rho] / float(rho + 1)
        out[b] = torch.sign(v[b]) * torch.clamp(u - theta, min=0.0)
    return out


def _project_halfspace_abs(
    w: torch.Tensor, d_hat: torch.Tensor, slack: torch.Tensor
) -> torch.Tensor:
    """Project onto |w·d̂| <= s (closed form)."""
    # d_hat: (B, K), slack: (B,)
    dot = (w * d_hat).sum(dim=-1, keepdim=True)
    s = slack.reshape(-1, 1)
    # If |dot| <= s: unchanged; else move along d_hat.
    excess = dot.abs() - s
    mask = (excess > 0).to(w.dtype)
    # sign(dot) * excess / ||d||^2 * d  (d unit ⇒ ||d||^2=1)
    corr = mask * torch.sign(dot) * excess * d_hat
    return w - corr


class ADMMProjectionLayer(nn.Module):
    """
    Approximate elastic CMDP via alternating projections (Dykstra/ADMM-style).

    Subproblems with known projes: box, ℓ₁-ball about w_prev, |w·Δ̂|≤s_δ.
    Slacks are chosen as soft residuals after hard projes, scaled by λ(σ)
    (high vol → larger effective s via lower λ pressure mirrored as larger s).

    Backward: straight-through / identity for first release — use cvxpy for
    exact grads when ``fallback`` or status-quo backend. Gradcheck gates live
    in tests; training with ADMM uses STE so policy grads still flow to actors.
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
        max_iters: int = 50,
        rho: float = 1.0,
        abs_tol: float = 1e-5,
        rel_tol: float = 1e-4,
        fallback_to_cvxpy: bool = True,
        use_ste: bool = True,
    ):
        super().__init__()
        self.use_ste = bool(use_ste)
        self.K = int(num_assets)
        self.tau0 = float(turnover_limit)
        self.tau = self.tau0
        self.penalty = float(penalty_weight)
        self.vol_scale_floor = float(vol_scale_floor)
        self.vol_scale_cap = float(vol_scale_cap)
        self.lambda_eps = float(lambda_eps)
        self.max_name = float(max_name_abs_weight)
        self.max_iters = int(max_iters)
        self.rho = float(rho)
        self.abs_tol = float(abs_tol)
        self.rel_tol = float(rel_tol)
        self.fallback_to_cvxpy = bool(fallback_to_cvxpy)
        self._cvx_oracle = None
        if self.fallback_to_cvxpy and self.K <= 50:
            from src.policy.convex_projection import ConvexProjectionLayer

            self._cvx_oracle = ConvexProjectionLayer(
                self.K,
                turnover_limit=self.tau0,
                penalty_weight=self.penalty,
                max_name_abs_weight=self.max_name,
            )

    def _resolve_vol(self, deltas, vol_scale):
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

    def forward(
        self,
        w_raw: torch.Tensor,
        w_prev: torch.Tensor,
        deltas: torch.Tensor,
        vol_scale: torch.Tensor | float | None = None,
        turnover_limit: torch.Tensor | float | None = None,
        return_slacks: bool = False,
    ):
        B, K = w_raw.shape
        device, dtype = w_raw.device, w_raw.dtype
        if turnover_limit is None:
            tau = torch.full((B,), self.tau0, device=device, dtype=dtype)
        elif isinstance(turnover_limit, (float, int)):
            tau = torch.full((B,), float(turnover_limit), device=device, dtype=dtype)
        else:
            tau = turnover_limit.detach().reshape(-1).to(device=device, dtype=dtype)
            if tau.numel() == 1 and B > 1:
                tau = tau.expand(B)

        d_scale = deltas.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        d_hat = deltas / d_scale
        vs = self._resolve_vol(deltas, vol_scale)
        # Soft delta slack budget grows as λ shrinks: s_δ ~ ||w_raw·d|| / (1+λ/λ0)
        lam = self.penalty / (vs + self.lambda_eps)
        s_delta = (w_raw * d_hat).sum(dim=-1).abs() / (1.0 + lam / self.penalty)
        s_delta = s_delta.clamp_min(0.0)

        w = w_raw
        u_box = torch.zeros_like(w)
        u_l1 = torch.zeros_like(w)
        u_hs = torch.zeros_like(w)
        rho = max(float(self.rho), 1e-6)
        for _ in range(self.max_iters):
            w_old = w
            # ADMM / Dykstra: project onto each set; dual scaled by ρ.
            z = w + u_box
            p = _project_box(z, self.max_name)
            u_box = u_box + rho * (w - p)
            w = p

            z = w + u_l1
            diff = _project_l1_ball(z - w_prev, tau)
            p = w_prev + diff
            u_l1 = u_l1 + rho * (w - p)
            w = p

            z = w + u_hs
            p = _project_halfspace_abs(z, d_hat, s_delta)
            u_hs = u_hs + rho * (w - p)
            w = p

            delta = (w - w_old).norm()
            scale = 1.0 + w.norm()
            if float(delta) < self.abs_tol + self.rel_tol * float(scale):
                break

        # Straight-through estimator (default): forward ADMM, backward identity.
        # When use_ste=False and cvxpy oracle available, return exact pathwise grads.
        if self.use_ste:
            w_out = w_raw + (w - w_raw).detach()
        elif self._cvx_oracle is not None:
            return self._cvx_oracle(
                w_raw,
                w_prev,
                deltas,
                vol_scale=vol_scale,
                turnover_limit=tau,
                return_slacks=return_slacks,
            )
        else:
            w_out = w  # no STE; grads may be broken through discrete projections

        # Optional residual check → cvxpy oracle for small K.
        if self._cvx_oracle is not None and self.fallback_to_cvxpy:
            resid = (w - w_raw).norm(dim=-1).mean()
            if float(resid) > 10.0:  # pathological
                return self._cvx_oracle(
                    w_raw,
                    w_prev,
                    deltas,
                    vol_scale=vol_scale,
                    turnover_limit=tau,
                    return_slacks=return_slacks,
                )

        s_turn = (w - w_prev).abs().sum(dim=-1) - tau
        s_turn = s_turn.clamp_min(0.0)
        if return_slacks:
            return w_out, s_delta, s_turn
        return w_out
