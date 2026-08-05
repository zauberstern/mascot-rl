"""AdaHedge, Follow-the-Leader, FlipFlop, and AdaHedge+Fixed-Share.

de Rooij et al. (2014): eta tuned from past mixability gap only (causal).
"""
from __future__ import annotations

import numpy as np

from mascotrl.eval.fixed_share import share_update


def _ftl_weights(S: np.ndarray, n: int) -> np.ndarray:
    m = float(np.min(S))
    mask = np.abs(S - m) <= 1e-15
    w = mask.astype(np.float64)
    s = float(w.sum())
    return w / s if s > 0 else np.ones(n, dtype=np.float64) / n


def _hedge_weights(S: np.ndarray, eta: float, n: int) -> np.ndarray:
    z = -float(eta) * S
    z = z - float(np.max(z))
    ex = np.exp(z)
    s = float(ex.sum())
    if s <= 0.0 or not np.isfinite(s):
        return np.ones(n, dtype=np.float64) / n
    return ex / s


def follow_the_leader(losses: np.ndarray) -> np.ndarray:
    """W[t] uniform on argmin of cumulative loss through t-1."""
    L = np.asarray(losses, dtype=np.float64)
    if L.ndim != 2:
        raise ValueError("losses must be (T, n)")
    t_len, n = L.shape
    if n < 2:
        raise ValueError("need at least 2 experts")
    S = np.zeros(n, dtype=np.float64)
    hist = np.zeros((t_len, n), dtype=np.float64)
    for t in range(t_len):
        hist[t] = _ftl_weights(S, n)
        S = S + L[t]
    return hist


def adahedge(losses: np.ndarray) -> np.ndarray:
    """AdaHedge: W[t] before seeing L[t]; eta from past Delta only."""
    L = np.asarray(losses, dtype=np.float64)
    if L.ndim != 2:
        raise ValueError("losses must be (T, n)")
    t_len, n = L.shape
    if n < 2:
        raise ValueError("need at least 2 experts")
    S = np.zeros(n, dtype=np.float64)
    Delta = 0.0
    hist = np.zeros((t_len, n), dtype=np.float64)
    log_n = float(np.log(n))
    for t in range(t_len):
        if Delta <= 0.0:
            w = _ftl_weights(S, n)
        else:
            eta = log_n / max(Delta, 1e-15)
            w = _hedge_weights(S, eta, n)
        hist[t] = w
        lt = L[t]
        if Delta <= 0.0:
            mix = float(np.min(lt))
        else:
            eta = log_n / max(Delta, 1e-15)
            z = -eta * lt
            z_max = float(np.max(z))
            mix = -(np.log(float(np.dot(w, np.exp(z - z_max)))) + z_max) / eta
        delta = max(0.0, float(np.dot(w, lt)) - mix)
        Delta += delta
        S = S + lt
    return hist


def adahedge_fixed_share(losses: np.ndarray, *, alpha: float) -> np.ndarray:
    """AdaHedge loss update then one Fixed-Share pool (shifting AdaHedge).

    Emits W[t] from the shared posterior carried from t-1; adapts eta via
    past mixability gap only.
    """
    L = np.asarray(losses, dtype=np.float64)
    if L.ndim != 2:
        raise ValueError("losses must be (T, n)")
    t_len, n = L.shape
    if n < 2:
        raise ValueError("need at least 2 experts")
    a = float(alpha)
    Delta = 0.0
    hist = np.zeros((t_len, n), dtype=np.float64)
    log_n = float(np.log(n))
    w = np.ones(n, dtype=np.float64) / n
    for t in range(t_len):
        hist[t] = w
        lt = L[t]
        if Delta <= 0.0:
            mix = float(np.min(lt))
            # eta → ∞: concentrate on instantaneous leaders among current support
            wm = w * np.exp(-1e6 * (lt - float(np.min(lt))))
        else:
            eta = log_n / max(Delta, 1e-15)
            z = -eta * lt
            z_max = float(np.max(z))
            mix = -(np.log(float(np.dot(w, np.exp(z - z_max)))) + z_max) / eta
            wm = w * np.exp(-eta * lt)
        delta = max(0.0, float(np.dot(w, lt)) - mix)
        Delta += delta
        s = float(wm.sum())
        if s <= 0.0 or not np.isfinite(s):
            wm = np.ones(n, dtype=np.float64) / n
        else:
            wm = wm / s
        w = share_update(wm, a)
    return hist


def flipflop(losses: np.ndarray) -> np.ndarray:
    """50/50 mix of AdaHedge and FTL weight vectors each day."""
    wa = adahedge(losses)
    wf = follow_the_leader(losses)
    mixed = 0.5 * wa + 0.5 * wf
    row = mixed.sum(axis=1, keepdims=True)
    row = np.where(row <= 0, 1.0, row)
    return mixed / row
