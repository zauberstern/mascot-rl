"""Log returns, price relatives, and 12-1 momentum from a return panel."""
from __future__ import annotations

import numpy as np

# Windows in trading days.
LOG_RETURN_WINDOWS: tuple[int, ...] = (1, 5, 21, 63, 126, 252)
MOM_12_1_LONG = 252
MOM_12_1_SKIP = 21


def _as_tk(returns: np.ndarray) -> np.ndarray:
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError(f"returns must be (T, K), got {r.shape}")
    return r


def cumulative_log_return(returns: np.ndarray, window: int) -> np.ndarray:
    """Causal sum of log(1+r) over ``window`` ending at each t (inclusive).

    Output ``(T, K)``; leading rows with incomplete history are NaN.
    """
    r = _as_tk(returns)
    t_len, k = r.shape
    w = int(window)
    if w <= 0:
        raise ValueError("window must be positive")
    log_r = np.log1p(np.clip(r, -0.999999, None))
    out = np.full((t_len, k), np.nan, dtype=np.float64)
    # Causal: at t use r[t-w+1 : t+1]
    csum = np.nancumsum(np.where(np.isfinite(log_r), log_r, 0.0), axis=0)
    for t in range(t_len):
        if t + 1 < w:
            continue
        start = t + 1 - w
        if start == 0:
            out[t] = csum[t]
        else:
            out[t] = csum[t] - csum[start - 1]
        # Invalidate if any missing in window.
        window_slice = log_r[start : t + 1]
        bad = ~np.all(np.isfinite(window_slice), axis=0)
        out[t, bad] = np.nan
    return out


def price_relative(returns: np.ndarray, window: int) -> np.ndarray:
    """``prod(1+r) - 1`` over causal window (simple cumulative return)."""
    r = _as_tk(returns)
    t_len, k = r.shape
    w = int(window)
    out = np.full((t_len, k), np.nan, dtype=np.float64)
    for t in range(t_len):
        if t + 1 < w:
            continue
        start = t + 1 - w
        block = r[start : t + 1]
        for j in range(k):
            col = block[:, j]
            if not np.all(np.isfinite(col)):
                continue
            out[t, j] = float(np.prod(1.0 + col) - 1.0)
    return out


def momentum_12_1(returns: np.ndarray) -> np.ndarray:
    """12-1 momentum: return over ~12m excluding most recent ~1m.

    At t: cum return over ``[t-252+1, t-21]`` (252-day long, skip last 21).
    """
    r = _as_tk(returns)
    t_len, k = r.shape
    out = np.full((t_len, k), np.nan, dtype=np.float64)
    long_w = MOM_12_1_LONG
    skip = MOM_12_1_SKIP
    for t in range(t_len):
        end = t - skip  # exclusive of recent month; inclusive end index = end
        start = t - long_w + 1
        if end < 0 or start < 0 or end < start:
            continue
        block = r[start : end + 1]
        for j in range(k):
            col = block[:, j]
            if not np.all(np.isfinite(col)) or col.size == 0:
                continue
            out[t, j] = float(np.prod(1.0 + col) - 1.0)
    return out


def residual_momentum_12_1(
    returns: np.ndarray,
    factors: np.ndarray,
    *,
    window: int = 252,
    skip: int = 21,
) -> np.ndarray:
    """12-1 momentum of FF residual returns (trailing OLS, causal).

    At each ``t``, fit ``r ~ factors`` on ``[t-window+1, t]`` (inclusive), take
    residuals, then cumulate residuals over ``[t-window+1, t-skip]``.
    """
    r = _as_tk(returns)
    f = np.asarray(factors, dtype=np.float64)
    if f.ndim != 2 or f.shape[0] != r.shape[0]:
        raise ValueError(f"factors must be (T, F) aligned with returns; got {f.shape}")
    t_len, k = r.shape
    n_f = int(f.shape[1])
    out = np.full((t_len, k), np.nan, dtype=np.float64)
    w = int(window)
    sk = int(skip)
    ones = np.ones((w, 1), dtype=np.float64)
    for t in range(t_len):
        start = t - w + 1
        end_skip = t - sk
        if start < 0 or end_skip < start:
            continue
        x = np.hstack([ones, f[start : t + 1]])
        if x.shape[0] < n_f + 2:
            continue
        for j in range(k):
            y = r[start : t + 1, j]
            if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
                continue
            try:
                coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            resid = y - x @ coef
            # Cumulate residual over [start, end_skip] relative to local window.
            local_end = end_skip - start
            if local_end < 0 or local_end >= resid.size:
                continue
            out[t, j] = float(np.sum(resid[: local_end + 1]))
    return out


def build_returns_momentum_block(
    returns: np.ndarray,
    *,
    factors: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Stack log-return windows, price relatives, mom 12-1 (+ residual mom)."""
    r = _as_tk(returns)
    channels: list[np.ndarray] = []
    names: list[str] = []
    for w in LOG_RETURN_WINDOWS:
        channels.append(cumulative_log_return(r, w))
        names.append(f"log_ret_{w}")
        channels.append(price_relative(r, w))
        names.append(f"price_rel_{w}")
    channels.append(momentum_12_1(r))
    names.append("mom_12_1")
    if factors is not None:
        channels.append(residual_momentum_12_1(r, factors))
        names.append("resid_mom_12_1")
    cube = np.stack(channels, axis=-1)
    return cube, names
