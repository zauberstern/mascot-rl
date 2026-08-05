"""Labeled experimental observation channels (not claim-eligible alphas)."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from src.features.blocks.range_volatility import parkinson_var_daily, _causal_mean
from src.features.blocks.volatility_vrp import trailing_hv_panel


def _as_tk(arr: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be (T,K), got {a.shape}")
    return a


def _empirical_tail_dep_asym(
    returns: np.ndarray,
    *,
    window: int = 63,
    threshold: float = 0.95,
) -> np.ndarray:
    """Per-asset upper minus lower empirical-copula tail dep vs EW market."""
    r = _as_tk(returns, "returns")
    t_len, k = r.shape
    out = np.full((t_len, k), np.nan, dtype=np.float64)
    w = int(window)
    for t in range(t_len):
        start = t - w + 1
        if start < 0:
            continue
        block = r[start : t + 1]
        mkt = np.nanmean(block, axis=1)
        if not np.all(np.isfinite(mkt)):
            finite_m = mkt[np.isfinite(mkt)]
            if finite_m.size < w // 2:
                continue
        for j in range(k):
            y = block[:, j]
            mask = np.isfinite(y) & np.isfinite(mkt)
            if mask.sum() < max(8, w // 3):
                continue
            yy, mm = y[mask], mkt[mask]
            n = yy.size
            # Average ranks → U(0,1)
            ry = (np.argsort(np.argsort(yy)).astype(float) + 1.0) / (n + 1.0)
            rm = (np.argsort(np.argsort(mm)).astype(float) + 1.0) / (n + 1.0)
            u = float(threshold)
            up_y, up_m = ry > u, rm > u
            lo_y, lo_m = ry < (1.0 - u), rm < (1.0 - u)
            def _lam(a: np.ndarray, b: np.ndarray) -> float:
                da = int(a.sum())
                if da == 0:
                    return float("nan")
                return float((a & b).sum()) / float(da)

            upper = _lam(up_y, up_m)
            lower = _lam(lo_y, lo_m)
            if np.isfinite(upper) and np.isfinite(lower):
                out[t, j] = upper - lower
    return out


def build_experimental_block(
    returns: np.ndarray,
    *,
    ohlc: Mapping[str, np.ndarray] | None = None,
    microstructure: Mapping[str, np.ndarray] | None = None,
    sentiment: Mapping[str, np.ndarray] | None = None,
    option_flow: Mapping[str, np.ndarray] | None = None,
    iv_surface: Mapping[str, np.ndarray] | None = None,
    gics_industry: Sequence[str | None] | None = None,
    mom_12_1: np.ndarray | None = None,
    amihud: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    r = _as_tk(returns, "returns")
    t_len, k = r.shape
    channels: list[np.ndarray] = []
    names: list[str] = []

    hv21 = trailing_hv_panel(r, 21)
    hv252 = trailing_hv_panel(r, 252)

    if ohlc is not None and all(x in ohlc for x in ("high", "low", "open", "close", "adj_close")):
        h = _as_tk(ohlc["high"], "high")
        l = _as_tk(ohlc["low"], "low")
        o = _as_tk(ohlc["open"], "open")
        c = _as_tk(ohlc["close"], "close")
        adj = _as_tk(ohlc["adj_close"], "adj_close")
        pk = _causal_mean(parkinson_var_daily(h, l), 21)
        with np.errstate(invalid="ignore"):
            pk_ann = np.sqrt(252.0 * np.maximum(pk, 0.0))
        ratio = pk_ann / np.where(hv21 == 0, np.nan, hv21)
        channels.append(ratio)
        names.append("x_range_cc_ratio")
        overnight = np.full((t_len, k), np.nan, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            overnight[1:] = np.log(o[1:] / adj[:-1])
        cc = np.full((t_len, k), np.nan, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            cc[1:] = np.log(c[1:] / c[:-1])
        on_m = _causal_mean(overnight, 21)
        cc_m = _causal_mean(np.abs(cc), 21)
        channels.append(on_m / np.where(cc_m == 0, np.nan, cc_m))
        names.append("x_overnight_share_21")

    if iv_surface is not None:
        mfiv30 = iv_surface.get("mfiv_30")
        mfiv365 = iv_surface.get("mfiv_365")
        if mfiv30 is not None and mfiv365 is not None:
            a = _as_tk(mfiv30, "mfiv_30")
            b = _as_tk(mfiv365, "mfiv_365")
            channels.append((a - hv21) - (b - hv252))
            names.append("x_vrp_term")

    if microstructure is not None and "eff_spread" in microstructure and amihud is not None:
        from src.features.blocks.microstructure import _causal_mean as _cm

        eff21 = _cm(_as_tk(microstructure["eff_spread"], "eff_spread"), 21)
        channels.append(eff21 * _as_tk(amihud, "amihud"))
        names.append("x_spread_x_amihud")

    if sentiment is not None and "si_pct" in sentiment and mom_12_1 is not None:
        channels.append(_as_tk(sentiment["si_pct"], "si_pct") * _as_tk(mom_12_1, "mom_12_1"))
        names.append("x_si_x_mom")

    if gics_industry is not None and len(gics_industry) == k:
        ret21 = np.full((t_len, k), np.nan, dtype=np.float64)
        for t in range(t_len):
            if t + 1 < 21:
                continue
            block = r[t + 1 - 21 : t + 1]
            with np.errstate(all="ignore"):
                ret21[t] = np.prod(1.0 + np.where(np.isfinite(block), block, 0.0), axis=0) - 1.0
                bad = ~np.all(np.isfinite(block), axis=0)
                ret21[t, bad] = np.nan
        ind = [str(x) if x is not None else "" for x in gics_industry]
        rel = np.full_like(ret21, np.nan)
        for t in range(t_len):
            row = ret21[t]
            for j in range(k):
                if not np.isfinite(row[j]) or not ind[j]:
                    continue
                peers = [row[i] for i in range(k) if ind[i] == ind[j] and np.isfinite(row[i])]
                if not peers:
                    continue
                rel[t, j] = row[j] - float(np.mean(peers))
        channels.append(rel)
        names.append("x_gics_rel_mom_21")

    if option_flow is not None and "oi_lvl" in option_flow and iv_surface is not None:
        oi = _as_tk(option_flow["oi_lvl"], "oi_lvl")
        oi_chg = np.full_like(oi, np.nan)
        oi_chg[21:] = oi[21:] - oi[:-21]
        skew = iv_surface.get("d_iv_skew_5d")
        if skew is not None:
            channels.append(oi_chg * _as_tk(skew, "d_iv_skew_5d"))
            names.append("x_oi_skew_flow")

    channels.append(_empirical_tail_dep_asym(r, window=63))
    names.append("x_tail_dep_asym_63")

    if not channels:
        return np.zeros((t_len, k, 0), dtype=np.float64), []
    return np.stack(channels, axis=-1), names
