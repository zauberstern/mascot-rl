"""Plot-only graph helpers.

Must never influence universe selection. Kept after the DII/TE retirement as a
diagnostic / densest-subgraph control visualiser.
"""
from __future__ import annotations

import numpy as np


def densest_subgraph_greedy(
    affinity: np.ndarray,
    k: int,
    *,
    seed_scores: np.ndarray | None = None,
) -> list[int]:
    """Greedily grow a set maximising affinity mass into the chosen set.

    Opposite of breadth-maximising selection. Plot / control only.
    """
    a = np.asarray(affinity, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("affinity must be square")
    d = a.shape[0]
    k = int(min(k, d))
    edge = a.copy()
    np.fill_diagonal(edge, 0.0)
    if seed_scores is None:
        scores = edge.sum(axis=1)
    else:
        scores = np.asarray(seed_scores, dtype=np.float64).reshape(-1)
    chosen = [int(np.argmax(scores))]
    remaining = [i for i in range(d) if i != chosen[0]]
    while len(chosen) < k and remaining:
        nxt = max(remaining, key=lambda i: float(edge[i, chosen].sum()))
        chosen.append(int(nxt))
        remaining.remove(nxt)
    return chosen
