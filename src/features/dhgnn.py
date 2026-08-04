"""Dynamic Hypergraph Neural Network with EMA copula / Pearson incidence."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pearson_tail_incidence(
    iv_history: torch.Tensor,
    threshold: float = 0.35,
) -> torch.Tensor:
    """Absolute Pearson corr of IV log-changes (levels if short), thresholded.

    iv_history: (T, K). Returns (K, K) soft scores in [0, 1] with diagonal 1.
    Peers below ``threshold`` are zeroed (universe_iv_corr_threshold style).
    """
    series = iv_history
    if series.dim() != 2:
        raise ValueError(f"iv_history must be (T, K); got shape={tuple(series.shape)}")
    T, K = series.shape
    device, dtype = series.device, series.dtype
    if T < 2 or K < 2:
        return torch.eye(K, device=device, dtype=dtype)

    # Prefer log-changes; fall back to levels when history is too short for diffs.
    if T >= 4:
        safe = series.clamp_min(1e-8)
        x = torch.log(safe[1:]) - torch.log(safe[:-1])
    else:
        x = series

    # Column-wise Pearson: corr = (Z^T Z) / (T'-1)
    mu = x.mean(dim=0, keepdim=True)
    sd = x.std(dim=0, keepdim=True, unbiased=True).clamp_min(1e-12)
    z = (x - mu) / sd
    n = max(int(z.shape[0]) - 1, 1)
    corr = (z.T @ z) / float(n)
    corr = 0.5 * (corr + corr.transpose(0, 1))
    abs_corr = corr.abs().clamp(0.0, 1.0)
    thr = float(threshold)
    scored = torch.where(abs_corr >= thr, abs_corr, torch.zeros_like(abs_corr))
    scored.fill_diagonal_(1.0)
    return scored


class SpatialDHGNN(nn.Module):
    """
    Cross-asset routing (spectrum spatial_mode).

    Default incidence H is rebuilt from an EMA of *empirical-copula upper-tail
    dependence* (not Pearson). Pearson is linear and lags crisis comovement;
    tail dependence λ_U = P(U_j > u | U_i > u) tracks nonlinear crash linkage
    that defines dispersion hyper-edges. `spatial_mode=dhgnn_pearson` selects
    absolute Pearson |corr| of IV log-changes; `none` disables spatial routing.

    Message passing follows spectral hypergraph convolution with learned
    hyper-edge weights W_e and equivariant filter Θ:
        X' = σ( D_v^{-1/2} H W_e D_e^{-1} Hᵀ D_v^{-1/2} X Θ )
    """

    def __init__(
        self,
        d_model: int,
        num_assets: int = 40,
        tail_threshold: float = 0.90,
        edge_threshold: float = 0.35,
        ema_alpha: float = 0.1,
        hist_len: int = 64,
        spatial_mode: str = "dhgnn_copula",
        update_incidence_at_eval: bool = False,
        allow_pearson_incidence: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_assets = num_assets
        self.tail_threshold = float(tail_threshold)
        self.edge_threshold = float(edge_threshold)
        self.ema_alpha = float(ema_alpha)
        self.hist_len = int(hist_len)
        self.update_incidence_at_eval = bool(update_incidence_at_eval)
        mode = str(spatial_mode).lower().strip()
        if mode in ("none", "off"):
            mode = "dhgnn_copula"  # disabled upstream via use_dhgnn=False
        if mode not in ("dhgnn_copula", "dhgnn_pearson"):
            raise ValueError(
                f"unknown spatial_mode={spatial_mode!r}; "
                "allowed=['dhgnn_copula', 'dhgnn_pearson']"
            )
        if mode == "dhgnn_pearson" and not bool(allow_pearson_incidence):
            raise ValueError(
                "spatial_mode=dhgnn_pearson requires allow_pearson_incidence=True "
                "(Pearson incidence is gated; default is empirical-copula upper-tail)"
            )
        self.spatial_mode = mode
        self.allow_pearson_incidence = bool(allow_pearson_incidence)
        self.pearson_incidence_used = mode == "dhgnn_pearson"

        # Equivariant operator Θ and per-edge gate (spectral We).
        self.theta = nn.Linear(d_model, d_model, bias=False)
        self.edge_gate = nn.Parameter(torch.ones(num_assets))
        self.vertex_proj = nn.Linear(d_model, d_model)
        self.out_norm = nn.LayerNorm(d_model)

        self.register_buffer(
            "running_tail", torch.eye(num_assets), persistent=True
        )
        self.register_buffer(
            "iv_hist", torch.zeros(hist_len, num_assets), persistent=True
        )
        self.register_buffer(
            "iv_hist_count", torch.zeros((), dtype=torch.long), persistent=True
        )

    def _resize_buffers(self, k: int, device: torch.device, dtype: torch.dtype) -> None:
        if self.running_tail.shape[0] != k:
            self.register_buffer(
                "running_tail",
                torch.eye(k, device=device, dtype=dtype),
                persistent=True,
            )
            self.register_buffer(
                "iv_hist",
                torch.zeros(self.hist_len, k, device=device, dtype=dtype),
                persistent=True,
            )
            self.iv_hist_count.zero_()
            # Keep edge_gate Parameter as-is; forward slices [:k].

    def _push_iv_history(self, iv_vec: torch.Tensor) -> None:
        """iv_vec: (K,) — append one snapshot into the ring buffer."""
        k = iv_vec.numel()
        n = int(self.iv_hist_count.item())
        slot = n % self.hist_len
        self.iv_hist[slot, :k] = iv_vec.detach()
        self.iv_hist_count += 1

    def observe_iv_step(self, iv_vec: torch.Tensor) -> None:
        """Append one env-step IV snapshot (PIT time series, not batch shuffles)."""
        self._push_iv_history(iv_vec)

    def _empirical_copula_tail_dependence(self, iv_series: torch.Tensor) -> torch.Tensor:
        """
        Upper-tail dependence matrix from empirical copula ranks.

        iv_series: (T, K) time history of IV levels (T≥2).
        Returns (K, K) with λ_U(i,j) on off-diagonals; diagonal = 1.
        """
        T, K = iv_series.shape
        if T < 4:
            return torch.eye(K, device=iv_series.device, dtype=iv_series.dtype)

        # Rank → U(0,1) marginals (average ranks for ties via argsort-argsort).
        order = torch.argsort(iv_series, dim=0)
        ranks = torch.argsort(order, dim=0).float() + 1.0
        U = ranks / float(T + 1)
        u = self.tail_threshold
        exceed = U > u  # (T, K)

        # λ_U(i,j) ≈ #{t: U_i>u and U_j>u} / #{t: U_i>u}
        joint = exceed.unsqueeze(2) & exceed.unsqueeze(1)  # (T, K, K)
        numer = joint.float().sum(dim=0)
        denom = exceed.float().sum(dim=0).clamp_min(1.0).unsqueeze(1)
        lam = numer / denom
        # Symmetrize for undirected hypergraph construction.
        lam = 0.5 * (lam + lam.transpose(0, 1))
        lam.fill_diagonal_(1.0)
        return lam.clamp(0.0, 1.0)

    def _instant_tail_proxy(self, iv_features: torch.Tensor) -> torch.Tensor:
        """
        Batch-wise fallback when history is short: rank assets cross-sectionally
        within the batch and form a soft co-exceedance proxy (still non-Pearson).
        """
        if iv_features.dim() == 2:
            x = iv_features
        else:
            x = iv_features.mean(dim=-1)
        b, k = x.shape
        # Cross-sectional ranks inside each batch row → weak copula proxy.
        order = torch.argsort(x, dim=-1)
        ranks = torch.argsort(order, dim=-1).float() + 1.0
        U = ranks / float(k + 1)
        u = max(0.7, self.tail_threshold - 0.1)
        exceed = U > u
        joint = exceed.unsqueeze(2) & exceed.unsqueeze(1)
        # If almost no exceedances, fall back to soft outer-product of U.
        soft = torch.einsum("bi,bj->bij", U, U)
        hard = joint.float()
        return torch.where(
            hard.sum(dim=(1, 2), keepdim=True) > 0,
            hard + 0.05 * soft,
            soft,
        ).clamp(0.0, 1.0)

    def _dependence_from_history(self, series: torch.Tensor) -> torch.Tensor:
        """Build (K, K) dependence from IV history under spatial_mode."""
        if self.spatial_mode == "dhgnn_pearson":
            return _pearson_tail_incidence(series, threshold=self.edge_threshold)
        return self._empirical_copula_tail_dependence(series)

    def _tail_matrix(self, iv_features: torch.Tensor) -> torch.Tensor:
        """EMA-smoothed (B, K, K) dependence used for incidence."""
        if iv_features.dim() == 2:
            iv = iv_features
        else:
            iv = iv_features.mean(dim=-1)
        b, k = iv.shape
        self._resize_buffers(k, iv.device, iv.dtype)

        # History updates only on explicit env-step snapshots or eval refresh.
        if self.update_incidence_at_eval:
            self._push_iv_history(iv.detach().mean(dim=0))

        n = int(self.iv_hist_count.item())
        if n >= 4:
            take = min(n, self.hist_len)
            # Oldest→newest ordering for ranks / log-changes.
            if n <= self.hist_len:
                series = self.iv_hist[:take]
            else:
                # Ring buffer: concatenate from ptr
                ptr = n % self.hist_len
                series = torch.cat(
                    [self.iv_hist[ptr:], self.iv_hist[:ptr]], dim=0
                )[-take:]
            hist_lam = self._dependence_from_history(series)
            if self.training or self.update_incidence_at_eval:
                self.running_tail.mul_(1.0 - self.ema_alpha).add_(
                    hist_lam, alpha=self.ema_alpha
                )
            # Temporal dependence is ready — do not mix cross-sectional ranks.
            return self.running_tail.unsqueeze(0).expand(b, -1, -1).clamp(0.0, 1.0)

        # Cold start only: soft cross-sectional proxy until T_hist ≥ 4.
        # Pearson mode still uses the copula cold-start proxy (no history yet).
        inst = self._instant_tail_proxy(iv)  # (B, K, K)
        a = self.ema_alpha
        return (a * inst + (1.0 - a) * self.running_tail.unsqueeze(0)).clamp(0.0, 1.0)

    def _build_dynamic_incidence_matrix(self, iv_features: torch.Tensor) -> torch.Tensor:
        """
        H: (B, K, E=K) — each asset anchors a hyper-edge with its top-2
        tail-co-exceeding peers (dispersion / index-basket structure).
        Soft weights = tail dependence scores (equivariant soft incidence).
        """
        tail = self._tail_matrix(iv_features)
        b, k, _ = tail.shape
        edges = []
        for i in range(k):
            scores = tail[:, i, :].clone()
            scores[:, i] = -1e9
            top2 = torch.topk(scores, k=min(2, k - 1), dim=-1)
            membership = torch.zeros(b, k, device=tail.device, dtype=tail.dtype)
            membership[:, i] = 1.0
            # Soft incidence from tail scores for selected peers.
            membership.scatter_(1, top2.indices, top2.values.clamp_min(self.edge_threshold))
            # Also admit any peer above hard tail threshold.
            membership = torch.maximum(
                membership, (scores > self.edge_threshold).to(tail.dtype) * scores.clamp_min(0)
            )
            membership[:, i] = 1.0
            edges.append(membership)
        return torch.stack(edges, dim=-1)  # (B, K, E)

    def forward(self, temporal_states: torch.Tensor, iv_features: torch.Tensor) -> torch.Tensor:
        # temporal_states: (B, K, D)
        H = self._build_dynamic_incidence_matrix(iv_features)  # (B, K, E)
        b, k, e = H.shape

        # Degree matrices
        deg_v = H.sum(dim=-1).clamp_min(1e-6)  # (B, K)
        deg_e = H.sum(dim=1).clamp_min(1e-6)   # (B, E)
        dv_inv_sqrt = deg_v.pow(-0.5)
        de_inv = deg_e.pow(-1.0)

        # Learned hyper-edge weights W_e (softplus for positivity).
        eg = self.edge_gate
        if eg.numel() < e:
            # Smoke / smaller K: tile
            eg = eg.repeat((e + eg.numel() - 1) // eg.numel())[:e]
        else:
            eg = eg[:e]
        we = F.softplus(eg).view(1, e, 1)

        # Spectral shift: D_v^{-1/2} H W_e D_e^{-1} Hᵀ D_v^{-1/2} X
        X = self.vertex_proj(temporal_states)
        X_s = X * dv_inv_sqrt.unsqueeze(-1)
        E = torch.einsum("bke,bkd->bed", H, X_s) * de_inv.unsqueeze(-1) * we
        X2 = torch.einsum("bke,bed->bkd", H, E) * dv_inv_sqrt.unsqueeze(-1)
        X2 = self.theta(X2)
        return self.out_norm(temporal_states + F.gelu(X2))
