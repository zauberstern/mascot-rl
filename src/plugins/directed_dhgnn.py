"""Directed hypergraph DHGNN (sibling to undirected SpatialDHGNN)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectedSpatialDHGNN(nn.Module):
    """
    Asymmetric tail routing via directed incidence H_tail / H_head.

    Estimates λ_U and λ_L without symmetrization. Each asset anchors a
    directed hyperarc: tail={i}, head=top-m peers by λ_U(i,·) (crash
    contagion out of i), plus lower-tail arcs from λ_L.
    """

    def __init__(
        self,
        d_model: int,
        num_assets: int = 40,
        tail_threshold: float = 0.90,
        lower_tail_threshold: float = 0.90,
        edge_threshold: float = 0.35,
        ema_alpha: float = 0.1,
        hist_len: int = 64,
        top_m: int = 2,
        laplace_alpha: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_assets = num_assets
        self.tail_threshold = float(tail_threshold)
        self.lower_tail_threshold = float(lower_tail_threshold)
        self.edge_threshold = float(edge_threshold)
        self.ema_alpha = float(ema_alpha)
        self.hist_len = int(hist_len)
        self.top_m = int(top_m)
        self.laplace_alpha = float(laplace_alpha)

        self.theta = nn.Linear(d_model, d_model, bias=False)
        self.edge_gate = nn.Parameter(torch.ones(num_assets * 2))
        self.vertex_proj = nn.Linear(d_model, d_model)
        self.out_norm = nn.LayerNorm(d_model)

        self.register_buffer("running_tail_u", torch.eye(num_assets), persistent=True)
        self.register_buffer("running_tail_l", torch.eye(num_assets), persistent=True)
        self.register_buffer(
            "iv_hist", torch.zeros(hist_len, num_assets), persistent=True
        )
        self.register_buffer(
            "iv_hist_count", torch.zeros((), dtype=torch.long), persistent=True
        )

    def _resize_buffers(self, k: int, device: torch.device, dtype: torch.dtype) -> None:
        if self.running_tail_u.shape[0] != k:
            self.register_buffer(
                "running_tail_u",
                torch.eye(k, device=device, dtype=dtype),
                persistent=True,
            )
            self.register_buffer(
                "running_tail_l",
                torch.eye(k, device=device, dtype=dtype),
                persistent=True,
            )
            self.register_buffer(
                "iv_hist",
                torch.zeros(self.hist_len, k, device=device, dtype=dtype),
                persistent=True,
            )
            self.iv_hist_count.zero_()

    def _push_iv_history(self, iv_vec: torch.Tensor) -> None:
        k = iv_vec.numel()
        n = int(self.iv_hist_count.item())
        slot = n % self.hist_len
        self.iv_hist[slot, :k] = iv_vec.detach()
        self.iv_hist_count += 1

    def _empirical_tail(
        self, iv_series: torch.Tensor, *, upper: bool
    ) -> torch.Tensor:
        T, K = iv_series.shape
        if T < 4:
            return torch.eye(K, device=iv_series.device, dtype=iv_series.dtype)
        order = torch.argsort(iv_series, dim=0)
        ranks = torch.argsort(order, dim=0).float() + 1.0
        U = ranks / float(T + 1)
        a = self.laplace_alpha
        if upper:
            u = self.tail_threshold
            exceed = U > u
        else:
            u = 1.0 - self.lower_tail_threshold
            exceed = U < u
        joint = exceed.unsqueeze(2) & exceed.unsqueeze(1)
        numer = joint.float().sum(dim=0) + a
        denom = exceed.float().sum(dim=0).clamp_min(0.0).unsqueeze(1) + 2.0 * a
        lam = numer / denom
        # Do NOT symmetrize — directed.
        lam.fill_diagonal_(1.0)
        return lam.clamp(0.0, 1.0)

    def _instant_tail_proxy(
        self, iv_features: torch.Tensor, *, upper: bool
    ) -> torch.Tensor:
        """Cross-sectional cold-start proxy (same spirit as undirected DHGNN)."""
        if iv_features.dim() == 2:
            x = iv_features
        else:
            x = iv_features.mean(dim=-1)
        b, k = x.shape
        order = torch.argsort(x, dim=-1)
        ranks = torch.argsort(order, dim=-1).float() + 1.0
        U = ranks / float(k + 1)
        if upper:
            u = max(0.7, self.tail_threshold - 0.1)
            exceed = U > u
        else:
            u = min(0.3, 1.0 - self.lower_tail_threshold + 0.1)
            exceed = U < u
        joint = exceed.unsqueeze(2) & exceed.unsqueeze(1)
        soft = torch.einsum("bi,bj->bij", U, U)
        hard = joint.float()
        out = torch.where(
            hard.sum(dim=(1, 2), keepdim=True) > 0,
            hard + 0.05 * soft,
            soft,
        ).clamp(0.0, 1.0)
        # Keep diagonal = 1; do NOT symmetrize (directed cold-start).
        eye = torch.eye(k, device=x.device, dtype=x.dtype).unsqueeze(0)
        return out * (1.0 - eye) + eye

    def _tail_matrices(self, iv_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if iv_features.dim() == 2:
            iv = iv_features
        else:
            iv = iv_features.mean(dim=-1)
        b, k = iv.shape
        self._resize_buffers(k, iv.device, iv.dtype)
        if self.training:
            self._push_iv_history(iv.detach().mean(dim=0))
        n = int(self.iv_hist_count.item())
        if n >= 4:
            take = min(n, self.hist_len)
            if n <= self.hist_len:
                series = self.iv_hist[:take]
            else:
                ptr = n % self.hist_len
                series = torch.cat(
                    [self.iv_hist[ptr:], self.iv_hist[:ptr]], dim=0
                )[-take:]
            lu = self._empirical_tail(series, upper=True)
            ll = self._empirical_tail(series, upper=False)
            if self.training:
                self.running_tail_u.mul_(1.0 - self.ema_alpha).add_(
                    lu, alpha=self.ema_alpha
                )
                self.running_tail_l.mul_(1.0 - self.ema_alpha).add_(
                    ll, alpha=self.ema_alpha
                )
            return (
                self.running_tail_u.unsqueeze(0).expand(b, -1, -1).clamp(0.0, 1.0),
                self.running_tail_l.unsqueeze(0).expand(b, -1, -1).clamp(0.0, 1.0),
            )
        # Cold start: soft cross-sectional proxy (not identity — avoids dead arcs).
        pu = self._instant_tail_proxy(iv, upper=True)
        pl = self._instant_tail_proxy(iv, upper=False)
        a = self.ema_alpha
        return (
            (a * pu + (1.0 - a) * self.running_tail_u.unsqueeze(0)).clamp(0.0, 1.0),
            (a * pl + (1.0 - a) * self.running_tail_l.unsqueeze(0)).clamp(0.0, 1.0),
        )
    def _build_directed_incidence(
        self, lam_u: torch.Tensor, lam_l: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns H_tail, H_head each (B, K, E) with E=2K (upper + lower arcs).
        """
        b, k, _ = lam_u.shape
        h_tail = []
        h_head = []
        m = min(self.top_m, max(k - 1, 1))
        for i in range(k):
            # Upper-tail arc: source i → top-m peers
            scores = lam_u[:, i, :].clone()
            scores[:, i] = -1e9
            top = torch.topk(scores, k=m, dim=-1)
            ht = torch.zeros(b, k, device=lam_u.device, dtype=lam_u.dtype)
            hh = torch.zeros(b, k, device=lam_u.device, dtype=lam_u.dtype)
            ht[:, i] = 1.0
            hh.scatter_(1, top.indices, top.values.clamp_min(self.edge_threshold))
            h_tail.append(ht)
            h_head.append(hh)
        for i in range(k):
            scores = lam_l[:, i, :].clone()
            scores[:, i] = -1e9
            top = torch.topk(scores, k=m, dim=-1)
            ht = torch.zeros(b, k, device=lam_l.device, dtype=lam_l.dtype)
            hh = torch.zeros(b, k, device=lam_l.device, dtype=lam_l.dtype)
            ht[:, i] = 1.0
            hh.scatter_(1, top.indices, top.values.clamp_min(self.edge_threshold))
            h_tail.append(ht)
            h_head.append(hh)
        return torch.stack(h_tail, dim=-1), torch.stack(h_head, dim=-1)

    def forward(self, temporal_states: torch.Tensor, iv_features: torch.Tensor) -> torch.Tensor:
        lam_u, lam_l = self._tail_matrices(iv_features)
        H_t, H_h = self._build_directed_incidence(lam_u, lam_l)
        b, k, e = H_t.shape

        deg_v = (H_t + H_h).sum(dim=-1).clamp_min(1e-6)
        deg_e = (H_t + H_h).sum(dim=1).clamp_min(1e-6)
        dv_inv_sqrt = deg_v.pow(-0.5)
        de_inv = deg_e.pow(-1.0)

        eg = self.edge_gate
        if eg.numel() < e:
            eg = eg.repeat((e + eg.numel() - 1) // eg.numel())[:e]
        else:
            eg = eg[:e]
        we = F.softplus(eg).view(1, e, 1)

        X = self.vertex_proj(temporal_states)
        # Message: tail → edge → head (directed).
        X_s = X * dv_inv_sqrt.unsqueeze(-1)
        E = torch.einsum("bke,bkd->bed", H_t, X_s) * de_inv.unsqueeze(-1) * we
        X2 = torch.einsum("bke,bed->bkd", H_h, E) * dv_inv_sqrt.unsqueeze(-1)
        X2 = self.theta(X2)
        return self.out_norm(temporal_states + F.gelu(X2))
