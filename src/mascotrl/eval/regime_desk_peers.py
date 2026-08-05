"""Causal rolling HRP / OLPS peers on a desk asset panel (eval only)."""
from __future__ import annotations

from typing import Any

import numpy as np

from mascotrl.eval.regime_desk_metrics import sharpe_annualized


def hrp_weights(cov: np.ndarray) -> np.ndarray:
    """Public thin wrapper around industry HRP on a covariance matrix."""
    from mascotrl.eval import industry_baselines as ib

    # Reconstruct via returns proxy: use cov as if from unit history is awkward;
    # call private bisection path with synthetic corr from cov.
    k = int(cov.shape[0])
    if k == 0:
        return np.zeros(0, dtype=np.float64)
    d = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / np.outer(d, d)
    corr = np.clip(np.nan_to_num(corr, nan=0.0), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    order = ib._hrp_cluster_order(corr)
    if len(order) != k:
        order = list(range(k))
    return ib._hrp_recursive_bisection(cov, order)


def causal_rolling_panel_returns(
    panel: np.ndarray,
    *,
    lookback: int = 252,
    min_obs: int = 60,
    mode: str = "hrp",
) -> dict[str, Any]:
    """Weights from history ending t-1 applied to return at t (no look-ahead).

    mode: "hrp" | "eg"
    """
    P = np.asarray(panel, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] < min_obs + 1 or P.shape[1] < 2:
        return {
            "returns": None,
            "sharpe": float("nan"),
            "limitation": "panel too thin for causal peer",
            "olps_stub_fallback": False,
        }
    t_len, k = P.shape
    out = np.full(t_len, np.nan, dtype=np.float64)
    w_prev = np.full(k, 1.0 / k, dtype=np.float64)
    stub = False
    for t in range(1, t_len):
        hist = P[max(0, t - lookback) : t]
        # Drop columns with too few finite obs
        finite_counts = np.isfinite(hist).sum(axis=0)
        keep = finite_counts >= min(min_obs, hist.shape[0])
        if int(keep.sum()) < 2:
            out[t] = float(np.nanmean(P[t]))
            continue
        H = hist[:, keep]
        H = np.where(np.isfinite(H), H, 0.0)
        rt = P[t]
        try:
            if mode == "hrp":
                if H.shape[0] < 2:
                    w_sub = np.full(H.shape[1], 1.0 / H.shape[1])
                else:
                    cov = np.cov(H, rowvar=False, ddof=1)
                    if cov.ndim == 0:
                        cov = np.array([[float(cov)]])
                    cov = np.nan_to_num(cov, nan=0.0)
                    # ridge for stability
                    cov = cov + np.eye(cov.shape[0]) * 1e-8
                    w_sub = hrp_weights(cov)
            else:
                from mascotrl.eval.olps import eg_weights, olps_weights

                try:
                    w_sub = olps_weights("eg", H)
                except Exception:
                    w_sub = eg_weights(H)
                    stub = True
            w_full = np.zeros(k, dtype=np.float64)
            w_full[np.where(keep)[0]] = w_sub
            s = float(w_full.sum())
            if s > 1e-15:
                w_full /= s
            else:
                w_full = w_prev
            w_prev = w_full
            r_t = np.where(np.isfinite(rt), rt, 0.0)
            out[t] = float(np.dot(w_full, r_t))
        except Exception:
            out[t] = float(np.nanmean(rt)) if np.isfinite(rt).any() else np.nan
    finite = out[np.isfinite(out)]
    if finite.size < min_obs:
        return {
            "returns": out,
            "sharpe": float("nan"),
            "limitation": "insufficient finite peer returns",
            "olps_stub_fallback": stub,
        }
    return {
        "returns": out,
        "sharpe": sharpe_annualized(out),
        "limitation": None,
        "olps_stub_fallback": stub,
    }
