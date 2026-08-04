"""Hypernetwork / DeepSets actor-critic backends (opt-in)."""
from __future__ import annotations

import torch
import torch.nn as nn


class HypernetActors(nn.Module):
    """
    Single hypernetwork generates per-asset actor weights from a conditioning
    embedding (asset id + optional state features). Status-quo remains ModuleList.
    """

    def __init__(
        self,
        num_assets: int,
        enriched_dim: int,
        embed_dim: int = 16,
        hidden: int = 64,
        condition_on: list[str] | None = None,
    ):
        super().__init__()
        self.num_assets = int(num_assets)
        self.enriched_dim = int(enriched_dim)
        self.hidden = int(hidden)
        # Non-empty condition_on → also condition on per-asset enriched state
        # (atm_iv / tail_centrality are carried inside E_i; we use the full vector).
        self.condition_on = list(condition_on or [])
        self.use_state_cond = len(self.condition_on) > 0
        self.embed = nn.Embedding(num_assets, embed_dim)
        n_w1 = enriched_dim * hidden
        n_b1 = hidden
        n_w2 = hidden * 1
        n_b2 = 1
        self.out_dim = n_w1 + n_b1 + n_w2 + n_b2
        hyper_in = embed_dim + (enriched_dim if self.use_state_cond else 0)
        self.hyper = nn.Sequential(
            nn.Linear(hyper_in, 64),
            nn.GELU(),
            nn.Linear(64, self.out_dim),
        )
        self._n_w1 = n_w1
        self._n_b1 = n_b1
        self._n_w2 = n_w2

    def _actor_means(self, enriched_states: torch.Tensor) -> torch.Tensor:
        # enriched: (B, K, D)
        B, K, D = enriched_states.shape
        means = []
        for i in range(K):
            emb = self.embed(
                torch.tensor(i % self.num_assets, device=enriched_states.device)
            )
            if self.use_state_cond:
                emb_b = emb.unsqueeze(0).expand(B, -1)
                hin = torch.cat([emb_b, enriched_states[:, i, :]], dim=-1)
                params = self.hyper(hin)  # (B, out_dim)
                off = 0
                w1 = params[:, off : off + self._n_w1].view(B, self.hidden, D)
                off += self._n_w1
                b1 = params[:, off : off + self._n_b1]
                off += self._n_b1
                w2 = params[:, off : off + self._n_w2].view(B, 1, self.hidden)
                off += self._n_w2
                b2 = params[:, off : off + 1]
                x = enriched_states[:, i, :].unsqueeze(1)  # (B,1,D)
                h = torch.nn.functional.gelu(
                    torch.bmm(x, w1.transpose(1, 2)).squeeze(1) + b1
                )
                means.append(
                    torch.bmm(h.unsqueeze(1), w2.transpose(1, 2)).squeeze(1) + b2
                )
            else:
                params = self.hyper(emb)
                off = 0
                w1 = params[off : off + self._n_w1].view(self.hidden, D)
                off += self._n_w1
                b1 = params[off : off + self._n_b1]
                off += self._n_b1
                w2 = params[off : off + self._n_w2].view(1, self.hidden)
                off += self._n_w2
                b2 = params[off : off + 1]
                x = enriched_states[:, i, :]
                h = torch.nn.functional.gelu(x @ w1.T + b1)
                means.append((h @ w2.T + b2))
        return torch.cat(means, dim=-1)


class DeepSetsCritic(nn.Module):
    """Permutation-equivariant pooling critic (K-agnostic input size)."""

    def __init__(self, enriched_dim: int, macro_dim: int, hidden: int = 128):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(enriched_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden + macro_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, enriched_states: torch.Tensor, macro_features: torch.Tensor
    ) -> torch.Tensor:
        h = self.phi(enriched_states).mean(dim=1)
        return self.rho(torch.cat([h, macro_features], dim=-1)).squeeze(-1)


class ModuleListActors(nn.Module):
    """Status-quo K independent MLPs (same architecture as HAPPOEngine.actors)."""

    def __init__(self, num_assets: int, enriched_dim: int):
        super().__init__()
        self.actors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(enriched_dim, 64),
                    nn.GELU(),
                    nn.Linear(64, 1),
                )
                for _ in range(num_assets)
            ]
        )

    def _actor_means(self, enriched_states: torch.Tensor) -> torch.Tensor:
        parts = [
            actor(enriched_states[:, i, :]) for i, actor in enumerate(self.actors)
        ]
        return torch.cat(parts, dim=-1)
