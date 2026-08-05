"""Range-based volatility estimators (Parkinson / Garman-Klass / Rogers-Satchell / Yang-Zhang)."""
from __future__ import annotations

import numpy as np

LN2 = float(np.log(2.0))
GK_CONST = 2.0 * LN2 - 1.0  # ≈ 0.386294361
ANN = float(np.sqrt(252.0))
DEFAULT_WINDOW = 21


def _as_tk(arr: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be (T, K), got {a.shape}")
    return a


def _causal_mean(panel: np.ndarray, window: int) -> np.ndarray:
    t_len, k = panel.shape
    w = int(window)
    out = np.full((t_len, k), np.nan, dtype=np.float64)
    for t in range(t_len):
        if t + 1 < w:
            continue
        block = panel[t + 1 - w : t + 1]
        for j in range(k):
            col = block[:, j]
            finite = col[np.isfinite(col)]
            if finite.size < max(2, w // 2):
                continue
            out[t, j] = float(np.mean(finite))
    return out


def _causal_var(panel: np.ndarray, window: int) -> np.ndarray:
    t_len, k = panel.shape
    w = int(window)
    out = np.full((t_len, k), np.nan, dtype=np.float64)
    for t in range(t_len):
        if t + 1 < w:
            continue
        block = panel[t + 1 - w : t + 1]
        for j in range(k):
            col = block[:, j]
            finite = col[np.isfinite(col)]
            if finite.size < max(2, w // 2):
                continue
            out[t, j] = float(np.var(finite, ddof=1))
    return out


def parkinson_var_daily(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    h = _as_tk(high, "high")
    l = _as_tk(low, "low")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.square(np.log(h / l)) / (4.0 * LN2)


def garman_klass_var_daily(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    o = _as_tk(open_, "open")
    h = _as_tk(high, "high")
    l = _as_tk(low, "low")
    c = _as_tk(close, "close")
    with np.errstate(divide="ignore", invalid="ignore"):
        return 0.5 * np.square(np.log(h / l)) - GK_CONST * np.square(np.log(c / o))


def rogers_satchell_var_daily(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    o = _as_tk(open_, "open")
    h = _as_tk(high, "high")
    l = _as_tk(low, "low")
    c = _as_tk(close, "close")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)


def yang_zhang_k(n: int) -> float:
    """Yang-Zhang weight: k = 0.34 / (1.34 + (n+1)/(n-1))."""
    n = int(n)
    if n <= 1:
        return 0.0
    return 0.34 / (1.34 + (n + 1) / (n - 1))


def build_range_volatility_block(
    ohlc: dict[str, np.ndarray] | None,
    *,
    window: int = DEFAULT_WINDOW,
) -> tuple[np.ndarray, list[str]]:
    """Build ``(T,K,C)`` range-vol block from extras['ohlc'] panels."""
    if ohlc is None:
        return np.zeros((0, 0, 0), dtype=np.float64), []
    required = ("open", "high", "low", "close")
    missing = [k for k in required if k not in ohlc]
    if missing:
        raise ValueError(f"ohlc missing keys {missing}")
    o = _as_tk(ohlc["open"], "open")
    h = _as_tk(ohlc["high"], "high")
    l = _as_tk(ohlc["low"], "low")
    c = _as_tk(ohlc["close"], "close")
    adj = _as_tk(ohlc.get("adj_close", c), "adj_close")
    t_len, k = o.shape
    w = int(window)
    pk = _causal_mean(parkinson_var_daily(h, l), w)
    gk = _causal_mean(garman_klass_var_daily(o, h, l, c), w)
    rs = _causal_mean(rogers_satchell_var_daily(o, h, l, c), w)
    # Overnight: ln(O_t / adj_close_{t-1})
    overnight = np.full((t_len, k), np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        overnight[1:] = np.log(o[1:] / adj[:-1])
    oc = np.full((t_len, k), np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        oc[:] = np.log(c / o)
    var_on = _causal_var(overnight, w)
    var_oc = _causal_var(oc, w)
    kk = yang_zhang_k(w)
    yz_var = var_on + kk * var_oc + (1.0 - kk) * rs
    with np.errstate(invalid="ignore"):
        channels = [
            ANN * np.sqrt(np.maximum(pk, 0.0)),
            ANN * np.sqrt(np.maximum(gk, 0.0)),
            ANN * np.sqrt(np.maximum(rs, 0.0)),
            ANN * np.sqrt(np.maximum(yz_var, 0.0)),
        ]
    cube = np.stack(channels, axis=-1)
    names = ["parkinson_21", "garman_klass_21", "rogers_satchell_21", "yang_zhang_21"]
    return cube, names
