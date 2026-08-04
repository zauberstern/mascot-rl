"""JKP / lottery / distributional cross-sectional features from returns + JKP panels."""
from __future__ import annotations

from typing import Mapping

import numpy as np

ANN = float(np.sqrt(252.0))


def _as_tk(arr: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be (T,K), got {a.shape}")
    return a


def max_ret(returns: np.ndarray, window: int = 21) -> np.ndarray:
    r = _as_tk(returns, "returns")
    t_len, k = r.shape
    w = int(window)
    out = np.full((t_len, k), np.nan, dtype=np.float64)
    for t in range(t_len):
        if t + 1 < w:
            continue
        block = r[t + 1 - w : t + 1]
        with np.errstate(all="ignore"):
            out[t] = np.nanmax(block, axis=0)
    return out


def min_ret(returns: np.ndarray, window: int = 21) -> np.ndarray:
    r = _as_tk(returns, "returns")
    t_len, k = r.shape
    w = int(window)
    out = np.full((t_len, k), np.nan, dtype=np.float64)
    for t in range(t_len):
        if t + 1 < w:
            continue
        block = r[t + 1 - w : t + 1]
        with np.errstate(all="ignore"):
            out[t] = np.nanmin(block, axis=0)
    return out


def ret_moment(returns: np.ndarray, window: int, moment: str) -> np.ndarray:
    r = _as_tk(returns, "returns")
    t_len, k = r.shape
    w = int(window)
    out = np.full((t_len, k), np.nan, dtype=np.float64)
    for t in range(t_len):
        if t + 1 < w:
            continue
        block = r[t + 1 - w : t + 1]
        for j in range(k):
            col = block[:, j]
            finite = col[np.isfinite(col)]
            if finite.size < max(4, w // 2):
                continue
            if moment == "skew":
                m = float(finite.mean())
                s = float(finite.std(ddof=1))
                out[t, j] = float(np.mean(((finite - m) / s) ** 3)) if s > 0 else np.nan
            else:
                m = float(finite.mean())
                s = float(finite.std(ddof=1))
                out[t, j] = float(np.mean(((finite - m) / s) ** 4)) if s > 0 else np.nan
    return out


def idio_vol_ff4(
    returns: np.ndarray,
    factors: np.ndarray,
    *,
    window: int = 63,
) -> np.ndarray:
    """Trailing OLS residual stdev * sqrt(252) on FF factors (Ang et al. 2006)."""
    r = _as_tk(returns, "returns")
    f = np.asarray(factors, dtype=np.float64)
    if f.ndim != 2 or f.shape[0] != r.shape[0]:
        raise ValueError(f"factors must be (T,F) aligned; got {f.shape}")
    t_len, k = r.shape
    w = int(window)
    n_f = int(f.shape[1])
    out = np.full((t_len, k), np.nan, dtype=np.float64)
    ones = np.ones((w, 1), dtype=np.float64)
    for t in range(t_len):
        start = t - w + 1
        if start < 0:
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
            if resid.size < 2:
                continue
            out[t, j] = float(np.std(resid, ddof=1) * ANN)
    return out


def beta_asym(
    returns: np.ndarray,
    market: np.ndarray,
    *,
    window: int = 63,
) -> np.ndarray:
    """Downside beta minus upside beta vs market (Ang-Chen-Xing 2006)."""
    r = _as_tk(returns, "returns")
    m = np.asarray(market, dtype=np.float64).reshape(-1)
    if m.shape[0] != r.shape[0]:
        raise ValueError("market must align with returns T")
    t_len, k = r.shape
    w = int(window)
    out = np.full((t_len, k), np.nan, dtype=np.float64)
    for t in range(t_len):
        start = t - w + 1
        if start < 0:
            continue
        m_win = m[start : t + 1]
        for j in range(k):
            y = r[start : t + 1, j]
            mask = np.isfinite(y) & np.isfinite(m_win)
            if mask.sum() < max(8, w // 3):
                continue
            yy, mm = y[mask], m_win[mask]
            down = mm < 0
            up = mm >= 0
            def _beta(ym: np.ndarray, xm: np.ndarray) -> float:
                if ym.size < 3 or float(np.var(xm)) < 1e-18:
                    return float("nan")
                return float(np.cov(ym, xm, ddof=1)[0, 1] / np.var(xm, ddof=1))

            bd = _beta(yy[down], mm[down]) if down.sum() >= 3 else float("nan")
            bu = _beta(yy[up], mm[up]) if up.sum() >= 3 else float("nan")
            if np.isfinite(bd) and np.isfinite(bu):
                out[t, j] = bd - bu
    return out


def build_jkp_lottery_block(
    returns: np.ndarray,
    *,
    jkp: Mapping[str, np.ndarray] | None = None,
    factors: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    r = _as_tk(returns, "returns")
    channels: list[np.ndarray] = [
        max_ret(r, 21),
        min_ret(r, 21),
        ret_moment(r, 63, "skew"),
        ret_moment(r, 63, "kurt"),
    ]
    names: list[str] = ["max_ret_21", "min_ret_21", "ret_skew_63", "ret_kurt_63"]
    if factors is not None:
        f = np.asarray(factors, dtype=np.float64)
        channels.append(idio_vol_ff4(r, f, window=63))
        names.append("idio_vol_ff4_21")
        # Market factor: first column of factors (Mkt-RF).
        mkt = f[:, 0] if f.ndim == 2 and f.shape[1] >= 1 else f.reshape(-1)
        channels.append(beta_asym(r, mkt, window=63))
        names.append("beta_asym_63")
    if jkp is not None:
        for key in ("log_me", "ivol_capm_21d", "ret_1_0"):
            if key in jkp:
                channels.append(_as_tk(jkp[key], key))
                names.append(key)
    cube = np.stack(channels, axis=-1)
    return cube, names
