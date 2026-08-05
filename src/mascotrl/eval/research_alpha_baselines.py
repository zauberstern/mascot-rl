"""Naive research baselines: random, equal-weight, Sign(lag return).

When factors + friction are supplied, baselines route through the parity
harness so Sharpes are comparable to the policy dual scorecard.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def _ann_sharpe(pnl: np.ndarray, *, periods: float = 252.0) -> float:
    x = np.asarray(pnl, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if sd <= 0.0 or not math.isfinite(sd):
        return float("nan")
    return float(math.sqrt(float(periods)) * mu / sd)


def random_baseline_sharpe(
    returns: np.ndarray, *, seed: int = 0, periods: float = 252.0
) -> float:
    """Single random L1 weight vector held constant (Zhang/Moody-style naive peer)."""
    rets = np.asarray(returns, dtype=np.float64)
    if rets.ndim != 2 or rets.shape[0] < 2:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    k = rets.shape[1]
    w = rng.normal(size=k)
    denom = float(np.sum(np.abs(w)))
    if denom <= 0.0:
        return float("nan")
    w = w / denom
    pnl = np.sum(w * np.nan_to_num(rets, nan=0.0), axis=1)
    return _ann_sharpe(pnl, periods=periods)


def equal_weight_sharpe(
    returns: np.ndarray,
    *,
    factors: np.ndarray | None = None,
    friction: Any | None = None,
    residualizer: Any | None = None,
    rebalance_mask: np.ndarray | None = None,
    cadence: str | None = None,
    scorecard: str = "total_net",
    periods: float = 252.0,
) -> float:
    """Equal-weight Sharpe.

    When ``factors`` and ``friction`` are provided, scores through the parity
    harness (costs + optional residualization). Otherwise falls back to gross
    ``w·r`` for backward-compatible unit tests.

    When ``rebalance_mask`` is supplied, ``cadence`` must be an explicit label
    (``daily`` / ``weekly`` / ``monthly``); the parity harness refuses density
    inference.
    """
    rets = np.asarray(returns, dtype=np.float64)
    if rets.ndim != 2 or rets.shape[0] < 2:
        return float("nan")
    if factors is not None and friction is not None:
        from src.eval.parity_harness import score_equal_weight

        out = score_equal_weight(
            rets,
            factors=np.asarray(factors, dtype=np.float64),
            friction=friction,
            residualizer=residualizer,
            rebalance_mask=rebalance_mask,
            cadence=cadence,
        )
        key = "residual" if scorecard == "residual" else "total_net"
        return _ann_sharpe(out[key], periods=periods)
    k = rets.shape[1]
    w = np.full(k, 1.0 / max(k, 1), dtype=np.float64)
    pnl = np.sum(w * np.nan_to_num(rets, nan=0.0), axis=1)
    return _ann_sharpe(pnl, periods=periods)


def long_baseline_sharpe(returns: np.ndarray, **kwargs: Any) -> float:
    """Zhang Long: equal long sleeve (same as EW on underlier panel)."""
    return equal_weight_sharpe(returns, **kwargs)


def sign_lag_return_sharpe(returns: np.ndarray, *, periods: float = 252.0) -> float:
    """Sign of prior-day cross-sectional mean return → long/flat sleeve."""
    rets = np.asarray(returns, dtype=np.float64)
    if rets.ndim != 2 or rets.shape[0] < 3:
        return float("nan")
    cross = np.nanmean(rets, axis=1)
    sig = np.sign(np.roll(cross, 1))
    sig[0] = 0.0
    k = rets.shape[1]
    w = (sig[:, None] / max(k, 1)) * np.ones((rets.shape[0], k), dtype=np.float64)
    pnl = np.sum(w * np.nan_to_num(rets, nan=0.0), axis=1)
    return _ann_sharpe(pnl, periods=periods)


def research_baselines_from_returns(
    returns: np.ndarray,
    *,
    seed: int = 0,
    factors: np.ndarray | None = None,
    friction: Any | None = None,
    residualizer: Any | None = None,
    rebalance_mask: np.ndarray | None = None,
    cadence: str | None = None,
    periods: float = 252.0,
) -> dict[str, dict[str, Any]]:
    kw = dict(
        factors=factors,
        friction=friction,
        residualizer=residualizer,
        rebalance_mask=rebalance_mask,
        cadence=cadence,
        periods=float(periods),
    )
    return {
        "random": {"sharpe": random_baseline_sharpe(returns, seed=seed, periods=periods)},
        "equal_weight": {"sharpe": equal_weight_sharpe(returns, **kw)},
        "long": {"sharpe": long_baseline_sharpe(returns, **kw)},
        "sign_lag": {"sharpe": sign_lag_return_sharpe(returns, periods=periods)},
    }


def policy_beats_peer(policy_sharpe: float, peer_sharpe: float) -> bool:
    try:
        p = float(policy_sharpe)
        r = float(peer_sharpe)
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(p) and math.isfinite(r)):
        return False
    return p > r


def policy_beats_random(policy_sharpe: float, random_sharpe: float) -> bool:
    return policy_beats_peer(policy_sharpe, random_sharpe)
