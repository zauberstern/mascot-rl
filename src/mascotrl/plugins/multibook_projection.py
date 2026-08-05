"""Multi-book block-batched CVXPY projection (exact IP per book, not joint K)."""
from __future__ import annotations

import torch
import torch.nn as nn

from mascotrl.policy.convex_projection import ConvexProjectionLayer


def partition_indices(
    n_assets: int,
    n_books: int,
    book_size: int,
    mode: str = "fixed_shards",
    scores: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """
    Return list of index tensors, one per book.

    Modes:
      - fixed_shards: contiguous blocks sized by ``book_size``. If
        ``n_books > 1`` and ``book_size >= n_assets``, shard into
        ``n_books`` nearly-equal contiguous blocks (so n_books is honored).
      - cluster_copula: greedy assign by row-sum of ``scores`` (K,K) tail matrix
      - sector_gics: requires ``scores`` as integer sector ids (K,)
    """
    K = int(n_assets)
    if mode == "fixed_shards":
        nb_cfg = max(1, int(n_books))
        bs = max(1, int(book_size))
        # Honor n_books when a single book_size would collapse to one shard.
        if nb_cfg > 1 and bs >= K:
            bs = max(1, (K + nb_cfg - 1) // nb_cfg)
        nb = max(1, (K + bs - 1) // bs)
        return [torch.arange(i * bs, min((i + 1) * bs, K)) for i in range(nb)]
    if mode == "sector_gics":
        if scores is None:
            raise ValueError("sector_gics partition needs sector id vector")
        ids = scores.reshape(-1).long()
        books = []
        for sid in torch.unique(ids).tolist():
            books.append(torch.where(ids == sid)[0])
        return books if books else [torch.arange(K)]
    if mode == "cluster_copula":
        # Greedy: sort by centrality, round-robin into n_books.
        nb = max(1, int(n_books))
        if scores is None:
            order = torch.arange(K)
        else:
            s = scores
            if s.dim() == 2:
                cent = s.sum(dim=-1)
            else:
                cent = s.reshape(-1)
            order = torch.argsort(cent, descending=True)
        books: list[list[int]] = [[] for _ in range(nb)]
        for i, idx in enumerate(order.tolist()):
            books[i % nb].append(idx)
        return [torch.tensor(b, dtype=torch.long) for b in books if b]
    raise ValueError(f"unknown multibook partition={mode!r}")


class MultibookProjectionLayer(nn.Module):
    """
    Apply ConvexProjectionLayer independently on index partitions.

    Joint |w·Δ| and global turnover are **not** enforced — disclose
    ``multi_book_block_projection`` in capital hygiene when enabled.
    """

    def __init__(
        self,
        num_assets: int,
        turnover_limit: float = 0.15,
        max_name_abs_weight: float = 5.0,
        n_books: int = 5,
        book_size: int = 50,
        partition: str = "fixed_shards",
        penalty_weight: float = 1e4,
        partition_scores: torch.Tensor | None = None,
    ):
        super().__init__()
        self.K = int(num_assets)
        self.tau0 = float(turnover_limit)
        self.tau = self.tau0
        self.max_name = float(max_name_abs_weight)
        self.n_books = int(n_books)
        self.book_size = int(book_size)
        self.partition = str(partition)
        self.penalty_weight = float(penalty_weight)
        self._init_scores = partition_scores
        self._layers = nn.ModuleDict()
        parts = partition_indices(
            self.K,
            self.n_books,
            self.book_size,
            self.partition,
            scores=partition_scores,
        )
        for p in parts:
            key = str(int(p.numel()))
            if key not in self._layers:
                self._layers[key] = ConvexProjectionLayer(
                    int(p.numel()),
                    turnover_limit=self.tau0,
                    penalty_weight=self.penalty_weight,
                    max_name_abs_weight=self.max_name,
                )
        self.register_buffer(
            "_part_meta",
            torch.zeros(1),
            persistent=False,
        )
        self._parts = parts

    def forward(
        self,
        w_raw: torch.Tensor,
        w_prev: torch.Tensor,
        deltas: torch.Tensor,
        vol_scale: torch.Tensor | float | None = None,
        turnover_limit: torch.Tensor | float | None = None,
        return_slacks: bool = False,
        partition_scores: torch.Tensor | None = None,
    ):
        B, K = w_raw.shape
        # Refresh partitions when scores are provided (cluster/sector) or K changes.
        needs_scores = self.partition in ("cluster_copula", "sector_gics")
        if K != self.K or (needs_scores and partition_scores is not None):
            scores = partition_scores if partition_scores is not None else self._init_scores
            parts = partition_indices(
                K, self.n_books, self.book_size, self.partition, scores
            )
        else:
            parts = self._parts
        w_out = w_raw.clone()
        s_delta_acc = []
        s_turn_acc = []
        for p in parts:
            p = p.to(device=w_raw.device)
            key = str(int(p.numel()))
            if key not in self._layers:
                self._layers[key] = ConvexProjectionLayer(
                    int(p.numel()),
                    turnover_limit=self.tau0,
                    penalty_weight=self.penalty_weight,
                    max_name_abs_weight=self.max_name,
                ).to(device=w_raw.device)
            layer = self._layers[key]
            # Per-book τ: scale by book_size/K so sum of budgets ≈ global τ.
            if turnover_limit is None:
                tau_b = self.tau0 * (float(p.numel()) / float(K))
            elif isinstance(turnover_limit, (float, int)):
                tau_b = float(turnover_limit) * (float(p.numel()) / float(K))
            else:
                tau_b = turnover_limit * (float(p.numel()) / float(K))
            wr = w_raw[:, p]
            wp = w_prev[:, p]
            d = deltas[:, p]
            out = layer(
                wr,
                wp,
                d,
                vol_scale=vol_scale,
                turnover_limit=tau_b,
                return_slacks=return_slacks,
            )
            if return_slacks:
                we, sd, st = out
                w_out[:, p] = we.to(dtype=w_out.dtype)
                s_delta_acc.append(sd)
                s_turn_acc.append(st)
            else:
                w_out[:, p] = out.to(dtype=w_out.dtype)
        # Honesty: joint |w·Δ| is not enforced by per-book QPs.
        self.last_joint_delta_residual = (w_out * deltas).sum(dim=-1).abs().detach()
        if return_slacks:
            if s_delta_acc:
                sd = torch.stack(s_delta_acc, dim=-1).sum(dim=-1)
                st = torch.stack(s_turn_acc, dim=-1).sum(dim=-1)
            else:
                sd = torch.zeros(B, device=w_raw.device, dtype=w_raw.dtype)
                st = torch.zeros(B, device=w_raw.device, dtype=w_raw.dtype)
            return w_out, sd, st
        return w_out

    @staticmethod
    def joint_neutrality_residual(
        w: torch.Tensor, deltas: torch.Tensor
    ) -> torch.Tensor:
        """|w·Δ| after multibook projection (expect ≥ single-book residual)."""
        return (w * deltas).sum(dim=-1).abs()
