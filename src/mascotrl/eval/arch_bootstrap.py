"""arch library adapters for stationary bootstrap (parity / optional path)."""
from __future__ import annotations

from typing import Any

import numpy as np

from mascotrl.eval.stats_rigor import annualized_sharpe, max_drawdown, stationary_bootstrap_indices
from mascotrl.logging_utils import get_logger

log = get_logger("mascotrl.eval.arch_bootstrap")


def stationary_bootstrap_indices_arch(
    n: int,
    *,
    block_mean: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """One stationary-bootstrap index path via ``arch.bootstrap.StationaryBootstrap``."""
    from arch.bootstrap import StationaryBootstrap

    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    x = np.arange(n, dtype=np.float64)
    bs = StationaryBootstrap(max(int(block_mean), 1), x, seed=int(seed))
    next(bs.bootstrap(1))
    idx = np.asarray(bs.index, dtype=np.int64).reshape(-1)
    if idx.size != n:
        raise RuntimeError(f"arch bootstrap index length {idx.size} != {n}")
    return idx


def block_bootstrap_metric_ci_arch(
    returns: np.ndarray | list[float],
    *,
    metric: str = "sharpe",
    n_boot: int = 499,
    block_mean: int = 5,
    alpha: float = 0.05,
    seed: int = 0,
    periods: int = 252,
) -> dict[str, Any]:
    """Same contract as ``block_bootstrap_metric_ci`` using arch index draws."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 10:
        return {"metric": metric, "n_obs": int(r.size), "point": float("nan"), "ci": None}

    def _compute(x: np.ndarray) -> float:
        if metric == "sharpe":
            return annualized_sharpe(x, periods=periods)
        if metric == "mean":
            return float(x.mean())
        if metric == "max_drawdown":
            return max_drawdown(x)
        if metric == "mean_abs":
            return float(np.mean(np.abs(x)))
        raise KeyError(metric)

    point = _compute(r)
    boots = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        idx = stationary_bootstrap_indices_arch(
            r.size, block_mean=block_mean, seed=int(seed) + b
        )
        boots[b] = _compute(r[idx])
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return {
        "metric": metric,
        "n_obs": int(r.size),
        "n_boot": int(n_boot),
        "block_mean": int(block_mean),
        "alpha": float(alpha),
        "point": float(point),
        "ci_low": lo,
        "ci_high": hi,
        "boot_mean": float(boots.mean()),
        "boot_std": float(boots.std(ddof=0)),
        "backend": "arch",
    }


def resolve_bootstrap_backend(cfg: dict[str, Any] | None = None) -> str:
    """``custom`` (default production) or ``arch``."""
    cfg = cfg or {}
    return str(cfg.get("bootstrap_backend") or "custom").lower().strip()


def stationary_bootstrap_indices_dispatch(
    n: int,
    *,
    block_mean: int = 5,
    rng: np.random.Generator | None = None,
    seed: int = 0,
    backend: str = "custom",
) -> np.ndarray:
    if str(backend).lower() == "arch":
        return stationary_bootstrap_indices_arch(n, block_mean=block_mean, seed=seed)
    return stationary_bootstrap_indices(n, block_mean=block_mean, rng=rng)
