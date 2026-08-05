"""Fold-fitted factor residualization for Alpha v2 rewards (FF4 / IPCA-3).

Reward identity (Phase 2 lock):
``residual = gross - costs - borrow - rf - lagged_exp · f_t``

Portfolio beta: rolling 252 trading-day OLS slopes through ``t-1`` (no intercept
in the reward subtraction). Asset E1 betas are a distinct object/hash field.
G5 uses calendar-month sums of daily residuals (no second residualization).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from mascotrl.logging_utils import get_logger

log = get_logger("mascotrl.eval.residualization")

# Distinct beta-object kinds for EstimandSpec / audit (must not collide).
PORTFOLIO_BETA_KIND = "portfolio_ff4_rolling_tminus1"
ASSET_E1_BETA_KIND = "asset_ff4_rolling_tminus1_e1"

# equity_substrate 7-factor layout: mkt,smb,hml,rmw,cma,umd[,ps_vwf].
# Classic FF4 residualization uses mkt,smb,hml,umd (skip rmw/cma/ps).
_FF4_FROM_WIDE_IDX = (0, 1, 2, 5)


def select_ff4_factor_matrix(X: Any) -> np.ndarray:
    """Return (T, 4) FF4 design from a 4-col or wider Gate2 factor panel."""
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)
    if X_arr.ndim != 2:
        raise ValueError(f"FF4 X must be 2-D; got shape {X_arr.shape}")
    n = int(X_arr.shape[1])
    if n == 4:
        return X_arr
    if n >= 6:
        return X_arr[:, list(_FF4_FROM_WIDE_IDX)]
    if n > 4:
        return X_arr[:, :4]
    raise ValueError(f"FF4 X must be (T, >=4); got shape {X_arr.shape}")


@dataclass(frozen=True)
class ResidualizerState:
    fold_id: str
    model: str
    betas: np.ndarray
    factor_names: tuple[str, ...]
    beta_kind: str = PORTFOLIO_BETA_KIND
    backend_used: str = "custom"


def fit_ff4_residualizer(
    y: Any,
    X: Any = None,
    *,
    asof_date: Any = None,
    fold_id: str = "fold",
    train_panel: Any = None,
    factor_returns: Mapping[str, np.ndarray] | None = None,
) -> ResidualizerState:
    """OLS betas of portfolio/asset returns on FF4 factors (train dates only).

    Primary API: ``y (T,)``, ``X (T, 4)`` or wider Gate2 panel ``(T, >=6)``
    from which classic FF4 columns (mkt,smb,hml,umd) are selected.
    Legacy kwargs ``train_panel`` / ``factor_returns`` remain for callers that
    pass named factor series.
    """
    del asof_date  # documented PIT boundary; caller must slice train dates
    names = ("mkt", "smb", "hml", "mom")
    if X is None and factor_returns is not None:
        cols = [np.asarray(factor_returns[n], dtype=np.float64).reshape(-1) for n in names]
        X = np.column_stack(cols)
        if y is None and train_panel is not None:
            y = train_panel
    if X is None:
        betas = np.zeros(len(names), dtype=np.float64)
    else:
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        X_arr = select_ff4_factor_matrix(X)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"y/X length mismatch {y_arr.shape[0]} vs {X_arr.shape[0]}"
            )
        betas, _, _, _ = np.linalg.lstsq(X_arr, y_arr, rcond=None)
        betas = np.asarray(betas, dtype=np.float64).reshape(4)
    return ResidualizerState(
        fold_id=str(fold_id),
        model="ff4",
        betas=betas,
        factor_names=names,
    )


def fit_ipca3_residualizer(
    train_option_panel: Any,
    characteristics: Any = None,
    *,
    asof_date: Any = None,
    fold_id: str = "fold",
    n_iter: int = 5,
    backend: str = "custom",
) -> ResidualizerState:
    """Fold-frozen 3-factor characteristic IPCA (promotion residual only).

    When ``characteristics`` is ``(T, N, L)`` or ``(N, L)``, runs a short
    alternating least-squares IPCA (Kelly–Pruitt / Gu–Kelly–Xiu style) on the
    train panel only. Without characteristics, falls back to truncated SVD
    PCA loadings (diagnostic, still fold-frozen).

    ``backend``:
      - ``custom`` (default): hand ALS / SVD
      - ``sklearn_pca``: sklearn TruncatedSVD when characteristics is None
      - ``ipca``: ``ipca.InstrumentedPCA`` when characteristics provided
        (falls back to custom on ImportError / shape failure)

    Promotion claim uses IPCA-3 residual only; other IPCA papers stay diagnostic.
    """
    del asof_date
    names = ("ipca1", "ipca2", "ipca3")
    backend = str(backend or "custom").lower().strip()
    panel = np.asarray(train_option_panel, dtype=np.float64)
    if panel.ndim == 1:
        panel = panel.reshape(-1, 1)
    if panel.size == 0:
        return ResidualizerState(
            fold_id=str(fold_id),
            model="ipca3",
            betas=np.zeros(3, dtype=np.float64),
            factor_names=names,
        )
    panel_c = panel - np.nanmean(panel, axis=0, keepdims=True)
    panel_c = np.nan_to_num(panel_c, nan=0.0)
    t_len, n_assets = int(panel_c.shape[0]), int(panel_c.shape[1])

    if characteristics is not None:
        char = np.asarray(characteristics, dtype=np.float64)
        if char.ndim == 2:
            # (N, L) static characteristics -> broadcast over T
            char = np.broadcast_to(char, (t_len, char.shape[0], char.shape[1]))
        if char.ndim != 3 or char.shape[0] != t_len or char.shape[1] != n_assets:
            raise ValueError(
                f"characteristics must be (T,N,L) or (N,L) matching panel; "
                f"got {char.shape} vs panel {(t_len, n_assets)}"
            )
        char = np.nan_to_num(char, nan=0.0)
        if backend == "ipca":
            lib_state = _fit_ipca3_library(
                panel_c, char, fold_id=fold_id, names=names
            )
            if lib_state is not None:
                return lib_state
        n_l = int(char.shape[2])
        # Initialize Gamma (L, 3) via SVD of mean characteristic-managed returns.
        # Managed portfolio: for each t, Z_t' r_t / N  -> (L,)
        managed = np.zeros((t_len, n_l), dtype=np.float64)
        for ti in range(t_len):
            managed[ti] = char[ti].T @ panel_c[ti] / max(n_assets, 1)
        managed -= managed.mean(axis=0, keepdims=True)
        _u, _s, vt = np.linalg.svd(managed, full_matrices=False)
        k = min(3, int(vt.shape[0]), n_l)
        gamma = np.zeros((n_l, 3), dtype=np.float64)
        gamma[:, :k] = vt[:k].T
        # ALS: factors f_t from Gamma' Z' r; update Gamma from stacked regressions.
        for _ in range(max(1, int(n_iter))):
            factors = np.zeros((t_len, 3), dtype=np.float64)
            for ti in range(t_len):
                z = char[ti]  # (N, L)
                beta_cs = z @ gamma  # (N, 3)
                # OLS r ~ beta_cs
                btb = beta_cs.T @ beta_cs + 1e-8 * np.eye(3)
                factors[ti] = np.linalg.solve(btb, beta_cs.T @ panel_c[ti])
            # Stack: r_{i,t} ≈ z_{i,t}' Gamma f_t
            # Vec form: for each (i,t), (f_t ⊗ z_{i,t})' vec(Gamma)
            rows = []
            ys = []
            for ti in range(t_len):
                ft = factors[ti]
                for i in range(n_assets):
                    zi = char[ti, i]
                    rows.append(np.kron(ft, zi))
                    ys.append(panel_c[ti, i])
            A = np.asarray(rows, dtype=np.float64)
            yv = np.asarray(ys, dtype=np.float64)
            gvec, _, _, _ = np.linalg.lstsq(A, yv, rcond=None)
            gamma = gvec.reshape(3, n_l).T  # (L, 3)
        # Asset loadings for portfolio exposure: mean Z @ Gamma over train
        mean_z = char.mean(axis=0)  # (N, L)
        loadings = mean_z @ gamma  # (N, 3)
        return ResidualizerState(
            fold_id=str(fold_id),
            model="ipca3",
            betas=np.asarray(loadings, dtype=np.float64),
            factor_names=names,
            backend_used="custom",
        )

    # Fallback: truncated SVD PCA loadings (fold-frozen diagnostic).
    if backend == "sklearn_pca":
        try:
            from sklearn.decomposition import TruncatedSVD

            k = min(3, n_assets, t_len)
            if k >= 1:
                svd = TruncatedSVD(n_components=k, algorithm="arpack", random_state=0)
                svd.fit(panel_c)
                loadings = np.asarray(svd.components_.T, dtype=np.float64)
                if k < 3:
                    loadings = np.hstack(
                        [
                            loadings,
                            np.zeros((loadings.shape[0], 3 - k), dtype=np.float64),
                        ]
                    )
                return ResidualizerState(
                    fold_id=str(fold_id),
                    model="ipca3",
                    betas=loadings,
                    factor_names=names,
                    backend_used="sklearn_pca",
                )
        except Exception as exc:
            log.warning("IPCA sklearn_pca backend fallback: %s", exc)
    _u, _s, vt = np.linalg.svd(panel_c, full_matrices=False)
    k = min(3, int(vt.shape[0]))
    loadings = vt[:k].T  # (N, k)
    if k < 3:
        loadings = np.hstack(
            [loadings, np.zeros((loadings.shape[0], 3 - k), dtype=np.float64)]
        )
    return ResidualizerState(
        fold_id=str(fold_id),
        model="ipca3",
        betas=np.asarray(loadings, dtype=np.float64),
        factor_names=names,
        backend_used="custom",
    )


def _fit_ipca3_library(
    panel_c: np.ndarray,
    char: np.ndarray,
    *,
    fold_id: str,
    names: tuple[str, ...],
) -> ResidualizerState | None:
    """Optional ``ipca.InstrumentedPCA`` path; None on failure."""
    try:
        from ipca import InstrumentedPCA
    except ImportError:
        return None
    t_len, n_assets, n_l = char.shape
    rows_x = []
    rows_y = []
    idx_e = []
    idx_t = []
    for ti in range(t_len):
        for i in range(n_assets):
            rows_x.append(char[ti, i])
            rows_y.append(panel_c[ti, i])
            idx_e.append(i)
            idx_t.append(ti)
    X = np.asarray(rows_x, dtype=np.float64)
    y = np.asarray(rows_y, dtype=np.float64)
    indices = np.column_stack(
        [np.asarray(idx_e, dtype=np.int64), np.asarray(idx_t, dtype=np.int64)]
    )
    try:
        regr = InstrumentedPCA(n_factors=3, intercept=False)
        regr = regr.fit(X=X, y=y, indices=indices)
        gamma = getattr(regr, "Gamma", None)
        if gamma is None:
            gamma, _factors = regr.get_factors(label_ind=False)
        gamma = np.asarray(gamma, dtype=np.float64)
        if gamma.ndim != 2:
            return None
        # Gamma is typically (L, K) or (K, L); normalize to (L, 3)
        if gamma.shape[0] == n_l:
            g = gamma[:, :3] if gamma.shape[1] >= 3 else gamma
        elif gamma.shape[1] == n_l:
            g = gamma[:3, :].T
        else:
            return None
        if g.shape[1] < 3:
            g = np.hstack([g, np.zeros((g.shape[0], 3 - g.shape[1]))])
        mean_z = char.mean(axis=0)
        loadings = mean_z @ g[:, :3]
        return ResidualizerState(
            fold_id=str(fold_id),
            model="ipca3",
            betas=np.asarray(loadings, dtype=np.float64),
            factor_names=names,
            backend_used="ipca",
        )
    except Exception as exc:
        log.warning("IPCA library backend fallback: %s", exc)
        return None


def residualize_step(
    gross_pnl: float,
    costs: float,
    exposures_tminus1: np.ndarray,
    factor_return_t: np.ndarray,
    *,
    borrow: float = 0.0,
    rf: float = 0.0,
) -> float:
    """reward = gross - costs - borrow - rf - lagged_exposure · factor_return.

    Daily RF is subtracted exactly once here (never again from Mkt-RF).
    """
    exp = np.asarray(exposures_tminus1, dtype=np.float64).reshape(-1)
    fac = np.asarray(factor_return_t, dtype=np.float64).reshape(-1)
    if exp.size != fac.size:
        raise ValueError(f"exposure/factor size mismatch {exp.size} vs {fac.size}")
    factor_pnl = float(np.dot(exp, fac))
    return (
        float(gross_pnl)
        - float(costs)
        - float(borrow)
        - float(rf)
        - factor_pnl
    )


def rolling_portfolio_ff4_beta(
    portfolio_returns: np.ndarray,
    factors: np.ndarray,
    *,
    t: int,
    window: int = 252,
) -> np.ndarray:
    """Portfolio-level FF4 slope betas from exactly ``window`` rows ending at t-1.

    Uses OLS with intercept for slope stability; returns the 4 slopes only
    (intercept is diagnostic, never subtracted from the reward). Raises if
    the window is incomplete (no partial-window fallback).
    """
    y = np.asarray(portfolio_returns, dtype=np.float64).reshape(-1)
    X = select_ff4_factor_matrix(factors)
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"y/X length mismatch {y.shape[0]} vs {X.shape[0]}")
    w = int(window)
    ti = int(t)
    if w <= 0:
        raise ValueError("window must be positive")
    if ti < w:
        raise ValueError(
            f"portfolio beta warmup incomplete: need t>={w} observations through "
            f"t-1; got t={ti}"
        )
    start = ti - w
    end = ti  # exclusive: rows [t-window, t-1]
    y_w = y[start:end]
    X_w = X[start:end]
    if y_w.shape[0] != w:
        raise ValueError(f"window length mismatch: got {y_w.shape[0]} want {w}")
    if not (np.isfinite(y_w).all() and np.isfinite(X_w).all()):
        raise ValueError("non-finite rows in portfolio beta window")
    # Design matrix with intercept; retain slopes only.
    ones = np.ones((w, 1), dtype=np.float64)
    A = np.concatenate([ones, X_w], axis=1)
    coef, _, _, _ = np.linalg.lstsq(A, y_w, rcond=None)
    return np.asarray(coef[1:5], dtype=np.float64).reshape(4)


def rolling_asset_ff4_residuals(
    returns: np.ndarray,
    factors: np.ndarray,
    *,
    t: int,
    window: int = 252,
    rf: np.ndarray | None = None,
) -> np.ndarray:
    """Per-asset FF4 residual panel through ``t-1`` (past-only asset betas).

    For each name i and each day s in ``[window, t)``, fit betas on
    ``returns[s-window:s, i]`` vs factors, then
    ``eps[s, i] = r[s,i] - rf[s] - beta' f[s]``.
    Returns shape ``(t, K)`` with leading warmup rows set to NaN.
    Distinct from :func:`rolling_portfolio_ff4_beta` (portfolio reward beta).
    """
    r = np.asarray(returns, dtype=np.float64)
    X = np.asarray(factors, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("returns must be (T, K)")
    if X.ndim != 2 or X.shape[1] < 4:
        raise ValueError(f"factors must be (T, >=4); got {X.shape}")
    X = X[:, :4]
    T, K = int(r.shape[0]), int(r.shape[1])
    ti = min(int(t), T)
    w = int(window)
    if rf is None:
        rf_arr = np.zeros(T, dtype=np.float64)
    else:
        rf_arr = np.asarray(rf, dtype=np.float64).reshape(-1)
        if rf_arr.size != T:
            raise ValueError(f"rf length {rf_arr.size} != T={T}")
    out = np.full((ti, K), np.nan, dtype=np.float64)
    if ti <= w or w <= 0:
        return out
    for s in range(w, ti):
        X_w = X[s - w : s]
        ones = np.ones((w, 1), dtype=np.float64)
        A = np.concatenate([ones, X_w], axis=1)
        fac_s = X[s]
        rf_s = float(rf_arr[s])
        for i in range(K):
            y_w = r[s - w : s, i]
            if not (np.isfinite(y_w).all() and np.isfinite(X_w).all()):
                continue
            coef, _, _, _ = np.linalg.lstsq(A, y_w, rcond=None)
            beta = coef[1:5]
            ri = float(r[s, i])
            if not np.isfinite(ri):
                continue
            out[s, i] = ri - rf_s - float(np.dot(beta, fac_s))
    return out


def aggregate_calendar_month_residuals(
    dates: Sequence[Any],
    daily_residuals: np.ndarray,
) -> pd.Series:
    """Calendar-month sum of daily residuals for G5 (no second residualization)."""
    idx = pd.to_datetime(list(dates))
    y = np.asarray(daily_residuals, dtype=np.float64).reshape(-1)
    if len(idx) != y.size:
        raise ValueError(f"dates/residuals length mismatch {len(idx)} vs {y.size}")
    s = pd.Series(y, index=idx, name="residual")
    return s.groupby(s.index.to_period("M")).sum()


def freeze_residualizer(state: ResidualizerState, fold_id: str) -> ResidualizerState:
    """Return an immutable copy tagged with fold_id (no refit after freeze)."""
    return ResidualizerState(
        fold_id=str(fold_id),
        model=state.model,
        betas=np.array(state.betas, dtype=np.float64, copy=True),
        factor_names=tuple(state.factor_names),
        beta_kind=getattr(state, "beta_kind", PORTFOLIO_BETA_KIND),
    )
