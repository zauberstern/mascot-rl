"""Industry-standard portfolio baselines (lookahead-safe).

All weight functions consume only ``returns_hist`` (train-window returns ending
at t-1). Never read future rows. NaN-safe; return zeros on insufficient data.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

INDUSTRY_BASELINE_NAMES = (
    "no_trade",
    "equal_weight",
    "ridge",
    "inverse_vol",
    "risk_parity_erc",
    "min_variance_lw",
    "hrp",
    "max_diversification",
    "mv_shrinkage",
    "xs_momentum_12_1",
    "short_term_reversal",
    "vol_managed",
    "buy_and_hold",
)

_MIN_OBS = 20
_EPS = 1e-12


def list_industry_baselines() -> tuple[str, ...]:
    return INDUSTRY_BASELINE_NAMES


def industry_baseline_weights(
    name: str,
    *,
    returns_hist: np.ndarray,
    t: int,
    w_prev: np.ndarray | None = None,
) -> np.ndarray:
    """Dispatch to a registered industry baseline.

    Parameters
    ----------
    name
        Key in ``INDUSTRY_BASELINE_REGISTRY``.
    returns_hist
        Shape ``(T_hist, K)`` train-window returns ending at decision date t-1.
    t
        Decision index in the full series (documentation / callers); not used
        to index into future returns.
    w_prev
        Previous weights (used by buy_and_hold).
    """
    try:
        fn = INDUSTRY_BASELINE_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown industry baseline: {name!r}") from exc
    return fn(returns_hist=returns_hist, t=t, w_prev=w_prev)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _as_hist(returns_hist: np.ndarray) -> np.ndarray:
    r = np.asarray(returns_hist, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("returns_hist must be (T_hist, K)")
    return r


def _zeros_k(k: int) -> np.ndarray:
    return np.zeros(k, dtype=np.float64)


def _equal_weight(k: int) -> np.ndarray:
    if k <= 0:
        return np.zeros(0, dtype=np.float64)
    return np.full(k, 1.0 / k, dtype=np.float64)


def _renorm_abs(w: np.ndarray) -> np.ndarray:
    s = float(np.nansum(np.abs(w)))
    if not np.isfinite(s) or s < _EPS:
        return _equal_weight(w.size)
    out = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    return out / s


def _trailing_vol(returns_hist: np.ndarray, *, min_obs: int = _MIN_OBS) -> np.ndarray:
    """Per-column nanstd; NaN where fewer than min_obs finite observations."""
    r = _as_hist(returns_hist)
    k = r.shape[1]
    vol = np.full(k, np.nan, dtype=np.float64)
    for j in range(k):
        col = r[:, j]
        finite = col[np.isfinite(col)]
        if finite.size >= min_obs:
            v = float(np.std(finite, ddof=1)) if finite.size > 1 else float(np.std(finite))
            vol[j] = v if np.isfinite(v) and v > 0 else np.nan
    return vol


def _sample_cov(returns_hist: np.ndarray) -> np.ndarray:
    """NaN-aware sample covariance with ridge fallback."""
    r = _as_hist(returns_hist)
    t_hist, k = r.shape
    if t_hist < 2 or k == 0:
        return np.eye(k, dtype=np.float64) * 1e-4

    # Column-wise demean with nanmean; fill NaN with 0 after demean.
    mu = np.nanmean(r, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    x = r - mu
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    # Count pairwise finite for a rough scale; use standard cov on filled data.
    cov = (x.T @ x) / max(t_hist - 1, 1)
    # Ridge for PSD / invertibility.
    diag = np.diag(cov).copy()
    diag = np.where(diag > _EPS, diag, 1e-4)
    ridge = 1e-6 * float(np.nanmean(diag))
    cov = cov + ridge * np.eye(k, dtype=np.float64)
    return cov


def _ledoit_wolf_cov(returns_hist: np.ndarray) -> np.ndarray:
    r = _as_hist(returns_hist)
    t_hist, k = r.shape
    if t_hist < 2 or k == 0:
        return _sample_cov(r)
    mu = np.nanmean(r, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    x = np.nan_to_num(r - mu, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf().fit(x)
        cov = np.asarray(lw.covariance_, dtype=np.float64)
        # Ensure finite + mild ridge.
        if not np.all(np.isfinite(cov)):
            return _sample_cov(r)
        ridge = 1e-8 * float(np.nanmean(np.diag(cov)))
        return cov + ridge * np.eye(k, dtype=np.float64)
    except Exception:
        return _sample_cov(r)


def _min_var_closed_form(cov: np.ndarray) -> np.ndarray:
    """Min w'Σw s.t. 1'w = 1 via closed form / least squares."""
    k = cov.shape[0]
    if k == 0:
        return np.zeros(0, dtype=np.float64)
    ones = np.ones(k, dtype=np.float64)
    try:
        # Solve Σ inv_ones = 1, then w = inv_ones / sum(inv_ones)
        inv_ones = np.linalg.solve(cov, ones)
    except np.linalg.LinAlgError:
        inv_ones, *_ = np.linalg.lstsq(cov, ones, rcond=None)
    s = float(np.sum(inv_ones))
    if not np.isfinite(s) or abs(s) < _EPS:
        return _equal_weight(k)
    w = inv_ones / s
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    # Clip extreme levered positions then renorm to sum=1.
    w = np.clip(w, -5.0, 5.0)
    s2 = float(np.sum(w))
    if abs(s2) < _EPS:
        return _equal_weight(k)
    return w / s2


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


def _no_trade(
    *,
    returns_hist: np.ndarray,
    t: int,
    w_prev: np.ndarray | None = None,
) -> np.ndarray:
    """G1 control: flat / cash (zero weights)."""
    del t, w_prev
    r = _as_hist(returns_hist)
    return _zeros_k(r.shape[1])


def _equal_weight_baseline(
    *,
    returns_hist: np.ndarray,
    t: int,
    w_prev: np.ndarray | None = None,
) -> np.ndarray:
    """G1 equal-weight 1/K across names in the hist panel."""
    del t, w_prev
    r = _as_hist(returns_hist)
    return _equal_weight(r.shape[1])


def _ridge(
    *,
    returns_hist: np.ndarray,
    t: int,
    w_prev: np.ndarray | None = None,
    ridge_lambda: float = 1e-2,
) -> np.ndarray:
    """G1 linear stub: ridge mean-variance on hist returns (same-feature baseline).

    Uses trailing mean as signal and ridge-shrunk cov; falls back to equal
    weight on singular / short panels.
    """
    del t, w_prev
    r = _as_hist(returns_hist)
    t_hist, k = r.shape
    if k == 0:
        return _zeros_k(0)
    if t_hist < _MIN_OBS:
        return _equal_weight(k)
    mu = np.nanmean(r, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    cov = _sample_cov(r)
    cov = cov + float(ridge_lambda) * np.eye(k, dtype=np.float64)
    try:
        raw = np.linalg.solve(cov, mu)
    except np.linalg.LinAlgError:
        raw, *_ = np.linalg.lstsq(cov, mu, rcond=None)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.sum(np.abs(raw))) < _EPS:
        return _equal_weight(k)
    return _renorm_abs(raw)


def _inverse_vol(
    *,
    returns_hist: np.ndarray,
    t: int,
    w_prev: np.ndarray | None = None,
) -> np.ndarray:
    del t, w_prev
    r = _as_hist(returns_hist)
    k = r.shape[1]
    if r.shape[0] < _MIN_OBS:
        return _zeros_k(k)
    vol = _trailing_vol(r)
    inv = np.where(np.isfinite(vol) & (vol > _EPS), 1.0 / vol, np.nan)
    if not np.any(np.isfinite(inv)):
        return _equal_weight(k)
    inv = np.nan_to_num(inv, nan=0.0)
    return _renorm_abs(inv)


def _risk_parity_erc(
    *,
    returns_hist: np.ndarray,
    t: int,
    w_prev: np.ndarray | None = None,
) -> np.ndarray:
    """Simple iterative ERC (Maillard-style) on vol estimates; fallback inverse_vol."""
    r = _as_hist(returns_hist)
    k = r.shape[1]
    if r.shape[0] < _MIN_OBS or k == 0:
        return _zeros_k(k)

    vol = _trailing_vol(r)
    if not np.any(np.isfinite(vol) & (vol > _EPS)):
        return _inverse_vol(returns_hist=r, t=t, w_prev=w_prev)

    # Start from inverse-vol; iterate toward equal risk contribution with diagonal
    # risk proxy (σ_i w_i). Full cov ERC is heavier; this matches the brief.
    w = _inverse_vol(returns_hist=r, t=t, w_prev=w_prev)
    sigma = np.nan_to_num(vol, nan=np.nanmedian(vol[np.isfinite(vol)]) if np.any(np.isfinite(vol)) else 1.0)
    sigma = np.maximum(sigma, _EPS)

    try:
        for _ in range(50):
            rc = w * sigma
            target = float(np.mean(rc))
            if target < _EPS:
                break
            # Update: w_i <- w_i * (target / rc_i)
            scale = np.where(rc > _EPS, target / rc, 1.0)
            w = w * scale
            w = _renorm_abs(w)
            if float(np.max(np.abs(rc - target))) < 1e-8:
                break
        if not np.all(np.isfinite(w)):
            return _inverse_vol(returns_hist=r, t=t, w_prev=w_prev)
        return w
    except Exception:
        return _inverse_vol(returns_hist=r, t=t, w_prev=w_prev)


def _min_variance_lw(
    *,
    returns_hist: np.ndarray,
    t: int,
    w_prev: np.ndarray | None = None,
) -> np.ndarray:
    del t, w_prev
    r = _as_hist(returns_hist)
    k = r.shape[1]
    if r.shape[0] < _MIN_OBS:
        return _zeros_k(k)
    cov = _ledoit_wolf_cov(r)
    return _min_var_closed_form(cov)


def _corr_distance(corr: np.ndarray) -> np.ndarray:
    c = np.clip(corr, -1.0, 1.0)
    d = np.sqrt(0.5 * (1.0 - c))
    np.fill_diagonal(d, 0.0)
    return d


def _greedy_seriation(dist: np.ndarray) -> list[int]:
    """Greedy nearest-neighbor order when scipy clustering is unavailable."""
    k = dist.shape[0]
    if k == 0:
        return []
    remaining = set(range(k))
    order = [0]
    remaining.remove(0)
    while remaining:
        last = order[-1]
        nxt = min(remaining, key=lambda j: dist[last, j])
        order.append(nxt)
        remaining.remove(nxt)
    return order


def _hrp_cluster_order(corr: np.ndarray) -> list[int]:
    dist = _corr_distance(corr)
    k = dist.shape[0]
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        # Condensed distance; clip tiny negatives from float error.
        condensed = squareform(dist, checks=False)
        condensed = np.maximum(condensed, 0.0)
        z = linkage(condensed, method="single")
        return list(int(i) for i in leaves_list(z))
    except Exception:
        return _greedy_seriation(dist) if k else []


def _hrp_recursive_bisection(cov: np.ndarray, order: list[int]) -> np.ndarray:
    """Lopez de Prado HRP recursive bisection on a seriated order."""
    k = cov.shape[0]
    w = np.ones(k, dtype=np.float64)
    clusters: list[list[int]] = [list(order)]
    while clusters:
        cluster = clusters.pop()
        if len(cluster) <= 1:
            continue
        mid = len(cluster) // 2
        left, right = cluster[:mid], cluster[mid:]
        # Cluster variance via inverse-variance within each side.
        def _cluster_var(idxs: list[int]) -> float:
            sub = cov[np.ix_(idxs, idxs)]
            iv = 1.0 / np.maximum(np.diag(sub), _EPS)
            iv = iv / np.sum(iv)
            return float(iv @ sub @ iv)

        v_l = _cluster_var(left)
        v_r = _cluster_var(right)
        alpha = 1.0 - v_l / (v_l + v_r + _EPS)
        for i in left:
            w[i] *= alpha
        for i in right:
            w[i] *= 1.0 - alpha
        if len(left) > 1:
            clusters.append(left)
        if len(right) > 1:
            clusters.append(right)
    s = float(np.sum(w))
    if s < _EPS:
        return _equal_weight(k)
    return w / s


def _hrp(
    *,
    returns_hist: np.ndarray,
    t: int,
    w_prev: np.ndarray | None = None,
) -> np.ndarray:
    del t, w_prev
    r = _as_hist(returns_hist)
    k = r.shape[1]
    if r.shape[0] < _MIN_OBS or k == 0:
        return _zeros_k(k)
    cov = _ledoit_wolf_cov(r)
    # Correlation from cov.
    d = np.sqrt(np.maximum(np.diag(cov), _EPS))
    corr = cov / np.outer(d, d)
    corr = np.clip(np.nan_to_num(corr, nan=0.0), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    order = _hrp_cluster_order(corr)
    if len(order) != k:
        order = list(range(k))
    return _hrp_recursive_bisection(cov, order)


def _max_diversification(
    *,
    returns_hist: np.ndarray,
    t: int,
    w_prev: np.ndarray | None = None,
) -> np.ndarray:
    """Max diversification: closed-form MDP on Σ (Choueifaty) with fallback."""
    del t, w_prev
    r = _as_hist(returns_hist)
    k = r.shape[1]
    if r.shape[0] < _MIN_OBS:
        return _zeros_k(k)
    cov = _ledoit_wolf_cov(r)
    vol = np.sqrt(np.maximum(np.diag(cov), _EPS))
    # MDP: max (w'σ) / sqrt(w'Σw) s.t. sum w = 1, w>=0 often.
    # Closed form proportional to Σ^{-1} σ (unconstrained long-only projection).
    try:
        inv_sig = np.linalg.solve(cov, vol)
    except np.linalg.LinAlgError:
        inv_sig, *_ = np.linalg.lstsq(cov, vol, rcond=None)
    w = np.maximum(inv_sig, 0.0)
    if float(np.sum(w)) < _EPS:
        # Practical fallback: inverse_vol adjusted by mean correlation.
        corr = cov / np.outer(vol, vol)
        mask = ~np.eye(k, dtype=bool)
        mean_corr = float(np.nanmean(corr[mask])) if k > 1 else 0.0
        mean_corr = mean_corr if np.isfinite(mean_corr) else 0.0
        adj = vol * max(abs(mean_corr), 0.1)
        inv = np.where(adj > _EPS, 1.0 / adj, 0.0)
        return _renorm_abs(inv)
    return _renorm_abs(w)


def _mv_shrinkage(
    *,
    returns_hist: np.ndarray,
    t: int,
    w_prev: np.ndarray | None = None,
) -> np.ndarray:
    """Mean-variance with LW/ridge Σ and risk aversion 1.0."""
    del t, w_prev
    r = _as_hist(returns_hist)
    k = r.shape[1]
    if r.shape[0] < _MIN_OBS:
        return _zeros_k(k)
    mu = np.nanmean(r, axis=0)
    mu = np.nan_to_num(mu, nan=0.0)
    cov = _ledoit_wolf_cov(r)
    risk_aversion = 1.0
    try:
        # w ∝ Σ^{-1} μ / λ
        raw = np.linalg.solve(cov, mu) / risk_aversion
    except np.linalg.LinAlgError:
        raw, *_ = np.linalg.lstsq(cov, mu, rcond=None)
        raw = raw / risk_aversion
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return _renorm_abs(raw)


def _xs_momentum_12_1(
    *,
    returns_hist: np.ndarray,
    t: int,
    w_prev: np.ndarray | None = None,
) -> np.ndarray:
    """12-1 cross-sectional momentum: cum return over last 252 excl. last 21."""
    del t, w_prev
    r = _as_hist(returns_hist)
    k = r.shape[1]
    t_hist = r.shape[0]
    if t_hist < 252:
        return _zeros_k(k)
    window = r[-252:]
    # Exclude last 21 days from the formation window.
    form = window[:-21]
    # Cumulative return via compounded product of (1+r); NaN-safe.
    safe = np.where(np.isfinite(form), form, 0.0)
    score = np.prod(1.0 + safe, axis=0) - 1.0
    # Assets with too few finite obs get NaN score → excluded.
    finite_counts = np.sum(np.isfinite(form), axis=0)
    score = np.where(finite_counts >= _MIN_OBS, score, np.nan)
    if not np.any(np.isfinite(score)):
        return _zeros_k(k)
    # Long top half / short bottom half equal weight among ranked finite.
    finite_idx = np.where(np.isfinite(score))[0]
    ranked = finite_idx[np.argsort(score[finite_idx])]
    n = ranked.size
    half = max(n // 2, 1)
    shorts = ranked[:half]
    longs = ranked[-half:]
    w = _zeros_k(k)
    if longs.size:
        w[longs] = 1.0 / longs.size
    if shorts.size:
        w[shorts] -= 1.0 / shorts.size
    return _renorm_abs(w)


def _short_term_reversal(
    *,
    returns_hist: np.ndarray,
    t: int,
    w_prev: np.ndarray | None = None,
) -> np.ndarray:
    """Score = -last 21d return; long high score (short recent winners)."""
    del t, w_prev
    r = _as_hist(returns_hist)
    k = r.shape[1]
    if r.shape[0] < 21:
        return _zeros_k(k)
    window = r[-21:]
    safe = np.where(np.isfinite(window), window, 0.0)
    ret_21 = np.prod(1.0 + safe, axis=0) - 1.0
    finite_counts = np.sum(np.isfinite(window), axis=0)
    ret_21 = np.where(finite_counts >= max(5, _MIN_OBS // 4), ret_21, np.nan)
    score = -ret_21
    if not np.any(np.isfinite(score)):
        return _zeros_k(k)
    finite_idx = np.where(np.isfinite(score))[0]
    ranked = finite_idx[np.argsort(score[finite_idx])]
    n = ranked.size
    half = max(n // 2, 1)
    shorts = ranked[:half]
    longs = ranked[-half:]
    w = _zeros_k(k)
    if longs.size:
        w[longs] = 1.0 / longs.size
    if shorts.size:
        w[shorts] -= 1.0 / shorts.size
    return _renorm_abs(w)


def _vol_managed(
    *,
    returns_hist: np.ndarray,
    t: int,
    w_prev: np.ndarray | None = None,
) -> np.ndarray:
    """Equal weight scaled by target_vol / realized_vol; clip scale to [0.25, 2]."""
    del t, w_prev
    r = _as_hist(returns_hist)
    k = r.shape[1]
    if r.shape[0] < _MIN_OBS or k == 0:
        return _zeros_k(k)
    # Portfolio equal-weight realized vol (daily), annualized.
    ew = _equal_weight(k)
    port = np.nansum(r * ew, axis=1)
    finite = port[np.isfinite(port)]
    if finite.size < _MIN_OBS:
        return _zeros_k(k)
    realized = float(np.std(finite, ddof=1)) * np.sqrt(252.0)
    if not np.isfinite(realized) or realized < _EPS:
        return ew
    target = 0.15
    scale = float(np.clip(target / realized, 0.25, 2.0))
    return ew * scale


def _buy_and_hold(
    *,
    returns_hist: np.ndarray,
    t: int,
    w_prev: np.ndarray | None = None,
) -> np.ndarray:
    del t
    r = _as_hist(returns_hist)
    k = r.shape[1]
    if w_prev is not None:
        prev = np.asarray(w_prev, dtype=np.float64).reshape(-1)
        if prev.size == k and np.all(np.isfinite(prev)):
            return prev.copy()
    return _equal_weight(k)


INDUSTRY_BASELINE_REGISTRY: dict[str, Callable[..., np.ndarray]] = {
    "no_trade": _no_trade,
    "equal_weight": _equal_weight_baseline,
    "ridge": _ridge,
    "inverse_vol": _inverse_vol,
    "risk_parity_erc": _risk_parity_erc,
    "min_variance_lw": _min_variance_lw,
    "hrp": _hrp,
    "max_diversification": _max_diversification,
    "mv_shrinkage": _mv_shrinkage,
    "xs_momentum_12_1": _xs_momentum_12_1,
    "short_term_reversal": _short_term_reversal,
    "vol_managed": _vol_managed,
    "buy_and_hold": _buy_and_hold,
}
