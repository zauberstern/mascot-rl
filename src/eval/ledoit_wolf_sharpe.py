"""Sharpe-ratio difference test for paired return series.

Literature note
---------------
Jobson–Korkie (1981) / Memmel (2003) provide an asymptotic paired Sharpe
difference z-test under i.i.d. (or weakly dependent) returns. Ledoit–Wolf
style covariance shrinkage appears elsewhere in this repo for *portfolio*
covariance, not for the Sharpe-difference sampling distribution.

Here we use a **stationary bootstrap of the Sharpe delta** (Politis & Romano
1994), reusing :func:`src.eval.stats_rigor.stationary_bootstrap_indices`.
That matches the repo's HAC-aware bootstrap style (SPA / metric CIs) and
avoids an iid Jobson–Korkie assumption on autocorrelated pnl. Method stamp:
``stationary_bootstrap_delta``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.eval.stats_rigor import annualized_sharpe, stationary_bootstrap_indices


def sharpe_difference_test(
    returns_a: np.ndarray | list[float],
    returns_b: np.ndarray | list[float],
    *,
    n_boot: int = 499,
    block_mean: int = 5,
    seed: int = 0,
    periods: float | int = 252,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Two-sided Sharpe difference test via stationary bootstrap of delta.

    Aligns on overlapping finite observations (truncated to common length,
    then drops pairwise non-finite rows). Returns point Sharpes, delta =
    SR_a - SR_b, bootstrap CI, and two-sided p-value under the null delta=0.
    """
    a = np.asarray(returns_a, dtype=np.float64).reshape(-1)
    b = np.asarray(returns_b, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n_obs = int(a.size)

    empty = {
        "sharpe_a": float("nan"),
        "sharpe_b": float("nan"),
        "delta": float("nan"),
        "pvalue": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "n_obs": n_obs,
        "n_boot": int(n_boot),
        "block_mean": int(block_mean),
        "method": "stationary_bootstrap_delta",
    }
    if n_obs < 10:
        return empty

    sr_a = annualized_sharpe(a, periods=periods)
    sr_b = annualized_sharpe(b, periods=periods)
    delta = float(sr_a - sr_b)

    rng = np.random.default_rng(int(seed))
    boots = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = stationary_bootstrap_indices(n_obs, block_mean=block_mean, rng=rng)
        boots[i] = annualized_sharpe(a[idx], periods=periods) - annualized_sharpe(
            b[idx], periods=periods
        )

    # Two-sided p-value: fraction of |boot - 0| as extreme as |delta| under
    # recentered null (boot_delta - mean(boot) + 0 ≈ boot - mean).
    # Simpler studentized-free: P(|delta*_c| >= |delta_obs|) with centering.
    center = float(np.mean(boots))
    centered = boots - center
    pvalue = float(np.mean(np.abs(centered) >= abs(delta) - 1e-15))
    # Keep p in [0, 1]; identical series → all centered ~0 → p≈1.
    pvalue = min(1.0, max(0.0, pvalue))

    lo = float(np.quantile(boots, float(alpha) / 2.0))
    hi = float(np.quantile(boots, 1.0 - float(alpha) / 2.0))

    return {
        "sharpe_a": float(sr_a),
        "sharpe_b": float(sr_b),
        "delta": delta,
        "pvalue": pvalue,
        "ci_low": lo,
        "ci_high": hi,
        "n_obs": n_obs,
        "n_boot": int(n_boot),
        "block_mean": int(block_mean),
        "method": "stationary_bootstrap_delta",
    }
