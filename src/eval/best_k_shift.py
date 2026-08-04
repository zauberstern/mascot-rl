"""Best-k-shift hindsight oracle (dynamic programming ceiling).

Formally defined tracking-the-best-expert optimum: among sequences with at
most ``k`` switches, minimize cumulative loss. No researcher cut-date choice.
"""
from __future__ import annotations

import math

import numpy as np


def best_k_shift(
    losses: np.ndarray,
    *,
    k: int,
) -> tuple[list[int], float]:
    """Return (expert_path, total_loss) for the best <=k-switch sequence.

    ``losses`` shape (T, n). Path length T. ``k=0`` is the best single expert;
    ``k=T-1`` is the omniscient per-step expert.
    """
    L = np.asarray(losses, dtype=np.float64)
    if L.ndim != 2:
        raise ValueError("losses must be (T, n)")
    t_len, n = L.shape
    if t_len == 0:
        return [], 0.0
    kk = int(k)
    if kk < 0:
        raise ValueError("k must be >= 0")
    kk = min(kk, t_len - 1)

    # dp[t, s, i] = min cost ending at t on expert i with exactly s switches.
    inf = np.inf
    dp = np.full((t_len, kk + 1, n), inf, dtype=np.float64)
    prev = np.full((t_len, kk + 1, n, 2), -1, dtype=np.int32)  # (s_prev, i_prev)

    dp[0, 0, :] = L[0]
    for t in range(1, t_len):
        for s in range(0, kk + 1):
            for i in range(n):
                # Stay on i.
                stay = dp[t - 1, s, i] + L[t, i]
                best = stay
                best_src = (s, i)
                # Switch from j != i using one switch budget.
                if s >= 1:
                    for j in range(n):
                        if j == i:
                            continue
                        cand = dp[t - 1, s - 1, j] + L[t, i]
                        if cand < best:
                            best = cand
                            best_src = (s - 1, j)
                dp[t, s, i] = best
                prev[t, s, i, 0] = best_src[0]
                prev[t, s, i, 1] = best_src[1]

    # Pick best ending (any s <= k, any i).
    end_s, end_i = 0, 0
    best_total = inf
    for s in range(0, kk + 1):
        for i in range(n):
            if dp[t_len - 1, s, i] < best_total:
                best_total = float(dp[t_len - 1, s, i])
                end_s, end_i = s, i

    path = [0] * t_len
    s, i = end_s, end_i
    for t in range(t_len - 1, -1, -1):
        path[t] = int(i)
        if t == 0:
            break
        s_prev = int(prev[t, s, i, 0])
        i_prev = int(prev[t, s, i, 1])
        s, i = s_prev, i_prev
    return path, float(best_total)


def theoretical_regret_bound(
    *,
    n_experts: int,
    k_switches: int,
    sequence_length: int,
) -> float:
    """O((k+1) ln n + k ln T) envelope (unit-constant form for reporting)."""
    n = int(n_experts)
    k = int(k_switches)
    t = int(sequence_length)
    if n < 2 or t < 1 or k < 0:
        raise ValueError("invalid bound args")
    return float((k + 1) * math.log(n) + k * math.log(max(t, 1)))
