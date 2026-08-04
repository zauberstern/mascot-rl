"""Sharpe returns-based style analysis (constrained non-negative loadings).

Interpretation only. Never feeds capital gates.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize


def fit_rbsa(
    portfolio_returns: np.ndarray | None,
    factor_returns: np.ndarray | None,
) -> tuple[np.ndarray, float]:
    """Constrained least squares: loadings >= 0, sum to 1.

    Returns ``(loadings, r_squared)``. Missing inputs yield all-NaN loadings
    and NaN R^2 so callers can stamp a data-availability reason.
    """
    if portfolio_returns is None or factor_returns is None:
        k = 0
        if factor_returns is not None:
            fac = np.asarray(factor_returns, dtype=np.float64)
            k = int(fac.shape[1]) if fac.ndim == 2 else 0
        return np.full(max(k, 1), np.nan), float("nan")

    r = np.asarray(portfolio_returns, dtype=np.float64).reshape(-1)
    f = np.asarray(factor_returns, dtype=np.float64)
    if f.ndim == 1:
        f = f.reshape(-1, 1)
    if r.size == 0 or f.size == 0 or r.shape[0] != f.shape[0]:
        return np.full(max(int(f.shape[1]) if f.ndim == 2 else 1, 1), np.nan), float("nan")

    # Drop rows with any NaN.
    mask = np.isfinite(r) & np.all(np.isfinite(f), axis=1)
    r = r[mask]
    f = f[mask]
    t, k = f.shape
    if t < k + 1 or k < 1:
        return np.full(k, np.nan), float("nan")

    def obj(c: np.ndarray) -> float:
        resid = r - f @ c
        return float(resid @ resid)

    x0 = np.full(k, 1.0 / k, dtype=np.float64)
    bounds = [(0.0, 1.0)] * k
    cons = {"type": "eq", "fun": lambda c: float(np.sum(c) - 1.0)}
    res = minimize(
        obj,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 200, "ftol": 1e-12},
    )
    loadings = np.asarray(res.x if res.success else x0, dtype=np.float64)
    loadings = np.clip(loadings, 0.0, None)
    s = float(loadings.sum())
    if s > 0:
        loadings = loadings / s
    else:
        loadings = x0
    resid = r - f @ loadings
    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((r - r.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else 0.0
    r2 = float(np.clip(r2, 0.0, 1.0))
    return loadings, r2


def rbsa_from_artifact(art: dict[str, Any]) -> dict[str, Any]:
    """Convenience: pull policy_returns / factors from a cell artifact."""
    port = art.get("policy_returns") or art.get("oos_returns")
    fac = art.get("factors") or art.get("oos_factors")
    names = list(art.get("factor_names") or [])
    if port is None or fac is None:
        return {
            "rbsa_loadings": [],
            "rbsa_r_squared": float("nan"),
            "factor_names": names,
            "data_availability_reason": "policy_returns_or_factors_missing",
        }
    loadings, r2 = fit_rbsa(np.asarray(port, dtype=np.float64), np.asarray(fac, dtype=np.float64))
    if not names and loadings.size:
        names = [f"f{i}" for i in range(int(loadings.shape[0]))]
    return {
        "rbsa_loadings": [float(x) for x in loadings],
        "rbsa_r_squared": float(r2),
        "factor_names": names,
    }
