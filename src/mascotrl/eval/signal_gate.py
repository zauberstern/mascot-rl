"""Signal IC gate: Fama-MacBeth / IC / allowlist (PIT fail-closed).

Selection window ends at ``selection_end`` (default 2012-12-31). Any date
passed for gating after that cutoff aborts. Empty allowlists fail closed via
``assert_allowlist_valid``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_EPS = 1e-12
_DEFAULT_SELECTION_END = "2012-12-31"
_ESTIMAND = "signal_ic_gate_v1"
_ESTIMAND_V2 = "signal_ic_gate_v2"
_ESTIMAND_GEOMETRY = "signal_obs_geometry_v1"
_HLZ_T_DEFAULT = 3.0
_FDR_Q_DEFAULT = 0.05
_LEGACY_T_MIN_DIAGNOSTIC = 2.0


def _as_panel(x: np.ndarray, name: str = "panel") -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be (T, K)")
    return a


def _date_str(d: Any) -> str:
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")
    s = str(d)
    # numpy datetime64 → '2012-12-31' or '2012-12-31T00:00:00'
    if "T" in s:
        s = s.split("T", 1)[0]
    return s[:10]


def _cs_ols_beta(y: np.ndarray, x: np.ndarray) -> float | None:
    """Cross-sectional OLS slope of y on x with intercept. None if underdetermined."""
    mask = np.isfinite(y) & np.isfinite(x)
    if int(mask.sum()) < 3:
        return None
    yy = y[mask]
    xx = x[mask]
    x_dm = xx - float(np.mean(xx))
    y_dm = yy - float(np.mean(yy))
    denom = float(np.dot(x_dm, x_dm))
    if denom <= _EPS:
        return None
    return float(np.dot(x_dm, y_dm) / denom)


def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    xr = x[mask]
    yr = y[mask]
    rx = np.empty_like(xr)
    ry = np.empty_like(yr)
    rx[np.argsort(xr)] = np.arange(xr.size, dtype=np.float64)
    ry[np.argsort(yr)] = np.arange(yr.size, dtype=np.float64)
    if float(np.std(rx)) < _EPS or float(np.std(ry)) < _EPS:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def fama_macbeth(
    signal_panel: np.ndarray,
    returns_panel: np.ndarray,
    *,
    lags: int = 1,
) -> dict[str, float]:
    """Cross-sectional FM regression of next-period returns on signal.

    At each date ``t``, regress ``returns[t+lags]`` on ``signal[t]`` with
    intercept. Returns mean slope, OLS t-stat across dates, and ``n_dates``.
    """
    out = fama_macbeth_detailed(signal_panel, returns_panel, lags=lags)
    return {
        "mean_coef": float(out["mean_coef"]),
        "t_stat": float(out["t_stat"]),
        "n_dates": int(out["n_dates"]),
    }


def fama_macbeth_detailed(
    signal_panel: np.ndarray,
    returns_panel: np.ndarray,
    *,
    lags: int = 1,
) -> dict[str, Any]:
    """FM slopes plus naive and Newey-West HAC t-stats on the slope series."""
    sig = _as_panel(signal_panel, "signal_panel")
    ret = _as_panel(returns_panel, "returns_panel")
    if sig.shape != ret.shape:
        raise ValueError("signal_panel and returns_panel must share shape (T, K)")
    lags_i = int(lags)
    if lags_i < 1:
        raise ValueError("lags must be >= 1")
    t_len = sig.shape[0]
    betas: list[float] = []
    for t in range(0, t_len - lags_i):
        b = _cs_ols_beta(ret[t + lags_i], sig[t])
        if b is not None and np.isfinite(b):
            betas.append(float(b))
    n = len(betas)
    if n == 0:
        return {
            "mean_coef": float("nan"),
            "t_stat": float("nan"),
            "t_stat_nw": float("nan"),
            "n_dates": 0,
            "slopes": np.asarray([], dtype=np.float64),
        }
    arr = np.asarray(betas, dtype=np.float64)
    mean_coef = float(np.mean(arr))
    if n < 2:
        t_stat = float("nan")
    else:
        se = float(np.std(arr, ddof=1)) / float(np.sqrt(n))
        t_stat = float(mean_coef / se) if se > _EPS else float("nan")
    from src.eval.stats_inference import hac_mean_tstat

    hac = hac_mean_tstat(arr)
    t_nw = float(hac.get("t_hac", float("nan")))
    return {
        "mean_coef": mean_coef,
        "t_stat": t_stat,
        "t_stat_nw": t_nw,
        "n_dates": int(n),
        "slopes": arr,
    }


def _two_sided_p_from_t(t_stat: float, *, n_dates: int) -> float:
    """Two-sided p-value for a t-stat (Student-t if df>=1 else normal)."""
    from scipy import stats

    t_abs = abs(float(t_stat))
    if not np.isfinite(t_abs):
        return float("nan")
    df = max(int(n_dates) - 1, 1)
    return float(2.0 * stats.t.sf(t_abs, df))


def decile_long_short(
    signal: np.ndarray,
    returns: np.ndarray,
    *,
    n_deciles: int = 10,
) -> dict[str, float]:
    """Long top / short bottom decile of signal; predictive (signal[t] → ret[t+1])."""
    sig = _as_panel(signal, "signal")
    ret = _as_panel(returns, "returns")
    if sig.shape != ret.shape:
        raise ValueError("signal and returns must share shape")
    n_d = int(n_deciles)
    if n_d < 2:
        raise ValueError("n_deciles must be >= 2")
    ls: list[float] = []
    for t in range(sig.shape[0] - 1):
        s = sig[t]
        r = ret[t + 1]
        mask = np.isfinite(s) & np.isfinite(r)
        if int(mask.sum()) < n_d:
            continue
        s_m = s[mask]
        r_m = r[mask]
        ranks = np.empty_like(s_m)
        ranks[np.argsort(s_m)] = np.arange(s_m.size, dtype=np.float64)
        pct = ranks / max(s_m.size - 1, 1)
        lo = pct <= (1.0 / n_d)
        hi = pct >= (1.0 - 1.0 / n_d)
        if not np.any(lo) or not np.any(hi):
            continue
        ls.append(float(np.mean(r_m[hi]) - np.mean(r_m[lo])))
    if not ls:
        return {"mean_return": float("nan"), "sharpe": float("nan")}
    arr = np.asarray(ls, dtype=np.float64)
    mean_r = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
    sharpe = float(mean_r / sd) if np.isfinite(sd) and sd > _EPS else float("nan")
    return {"mean_return": mean_r, "sharpe": sharpe}


def ic_series(signal: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Spearman rank IC of signal[t] vs returns[t+1] for each date."""
    sig = _as_panel(signal, "signal")
    ret = _as_panel(returns, "returns")
    if sig.shape != ret.shape:
        raise ValueError("signal and returns must share shape")
    out = np.full(sig.shape[0] - 1, np.nan, dtype=np.float64)
    for t in range(out.size):
        out[t] = _spearman_ic(sig[t], ret[t + 1])
    return out


def ic_decay(
    signal: np.ndarray,
    returns: np.ndarray,
    horizons: Sequence[int] = (1, 3, 6, 12),
) -> dict[int, float]:
    """Mean Spearman IC at each forward horizon (panel steps)."""
    sig = _as_panel(signal, "signal")
    ret = _as_panel(returns, "returns")
    if sig.shape != ret.shape:
        raise ValueError("signal and returns must share shape")
    out: dict[int, float] = {}
    for h in horizons:
        hh = int(h)
        if hh < 1:
            raise ValueError("horizons must be >= 1")
        ics: list[float] = []
        for t in range(0, sig.shape[0] - hh):
            v = _spearman_ic(sig[t], ret[t + hh])
            if np.isfinite(v):
                ics.append(float(v))
        out[hh] = float(np.mean(ics)) if ics else float("nan")
    return out


def ff_alpha(
    y: np.ndarray,
    factors: np.ndarray,
    *,
    lags: int | None = None,
) -> dict[str, Any]:
    """Newey-West HAC alpha of ``y`` (e.g. decile long-short returns) on
    ``factors`` (e.g. FF4 + Pastor-Stambaugh traded liquidity).

    OLS with a Bartlett-kernel HAC sandwich covariance matrix (Newey and
    West, 1987) on the intercept, so serially correlated monthly overlap
    does not silently overstate significance.
    """
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    xx = np.asarray(factors, dtype=np.float64)
    if xx.ndim == 1:
        xx = xx.reshape(-1, 1)
    if xx.shape[0] != yy.size:
        raise ValueError(f"factors rows {xx.shape[0]} != y length {yy.size}")
    mask = np.isfinite(yy) & np.all(np.isfinite(xx), axis=1)
    yy = yy[mask]
    xx = xx[mask]
    n = int(yy.size)
    p = int(xx.shape[1]) + 1
    if n < p + 2:
        return {"alpha": float("nan"), "t_stat": float("nan"), "n": n, "lags": 0}
    design = np.column_stack([np.ones(n), xx])
    beta, *_ = np.linalg.lstsq(design, yy, rcond=None)
    resid = yy - design @ beta
    xtx_inv = np.linalg.pinv(design.T @ design)
    from src.eval.stats_inference import newey_west_lag

    l_bw = int(newey_west_lag(n) if lags is None else max(0, int(lags)))
    scores = design * resid[:, None]
    s_mat = scores.T @ scores
    for j in range(1, l_bw + 1):
        if j >= n:
            break
        w = 1.0 - j / (l_bw + 1.0)
        gamma_j = scores[j:].T @ scores[:-j]
        s_mat = s_mat + w * (gamma_j + gamma_j.T)
    cov = xtx_inv @ s_mat @ xtx_inv
    se_alpha = float(np.sqrt(max(cov[0, 0], 0.0)))
    alpha = float(beta[0])
    t_stat = float(alpha / se_alpha) if se_alpha > _EPS else float("nan")
    return {"alpha": alpha, "t_stat": t_stat, "n": n, "lags": l_bw}


def effective_breadth(corr_matrix: np.ndarray) -> float:
    """ENB = (sum λ)² / sum(λ²) from correlation eigenvalues."""
    c = np.asarray(corr_matrix, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] < 1:
        return float("nan")
    c = 0.5 * (c + c.T)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        evals = np.linalg.eigvalsh(c)
    except np.linalg.LinAlgError:
        return float("nan")
    evals = np.clip(evals, 0.0, None)
    s1 = float(evals.sum())
    s2 = float(np.sum(evals * evals))
    if s2 <= _EPS or not np.isfinite(s1) or not np.isfinite(s2):
        return float("nan")
    return float((s1 * s1) / s2)


def orthogonalize_signals(
    signals: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Sequential cross-sectional residualization (Gram-Schmidt) in key order."""
    if not signals:
        return {}
    names = list(signals.keys())
    panels = [_as_panel(signals[n], n) for n in names]
    shape0 = panels[0].shape
    for n, p in zip(names, panels):
        if p.shape != shape0:
            raise ValueError(f"signal {n!r} shape {p.shape} != {shape0}")
    t_len, _k = shape0
    out_panels = [np.full_like(p, np.nan) for p in panels]
    for t in range(t_len):
        cols: list[np.ndarray] = []
        for i, p in enumerate(panels):
            x = p[t].copy()
            for prev in cols:
                mask = np.isfinite(x) & np.isfinite(prev)
                if int(mask.sum()) < 3:
                    continue
                b = _cs_ols_beta(x, prev)
                if b is None:
                    continue
                a = float(np.nanmean(x[mask] - b * prev[mask]))
                x = x - (a + b * prev)
            out_panels[i][t] = x
            cols.append(x)
    return {n: out_panels[i] for i, n in enumerate(names)}


def _signal_corr_matrix(signals: Mapping[str, np.ndarray]) -> np.ndarray:
    names = list(signals.keys())
    if not names:
        return np.zeros((0, 0), dtype=np.float64)
    flat = []
    for n in names:
        flat.append(np.nan_to_num(_as_panel(signals[n], n), nan=0.0).ravel())
    mat = np.vstack(flat)
    if mat.shape[0] == 1:
        return np.array([[1.0]], dtype=np.float64)
    c = np.corrcoef(mat)
    return np.asarray(c, dtype=np.float64)


def run_signal_gate(
    signals: Mapping[str, np.ndarray],
    returns: np.ndarray,
    *,
    dates: Sequence[Any],
    selection_end: str = _DEFAULT_SELECTION_END,
    t_min: float = 2.0,
) -> dict[str, Any]:
    """Admit signals with |FM t_stat| > t_min on the PIT selection window.

    Fail-closed: any date in ``dates`` strictly after ``selection_end`` raises.
    """
    ret = _as_panel(returns, "returns")
    date_list = list(dates)
    if len(date_list) != ret.shape[0]:
        raise ValueError(
            f"dates length {len(date_list)} != returns T={ret.shape[0]}"
        )
    end = _date_str(selection_end)
    parsed = [_date_str(d) for d in date_list]
    bad = [d for d in parsed if d > end]
    if bad:
        raise ValueError(
            f"signal gate refuses dates after selection_end={end}: "
            f"found {bad[0]} (+{len(bad) - 1} more)"
        )

    stats: dict[str, dict[str, Any]] = {}
    allowlist: list[str] = []
    admitted_panels: dict[str, np.ndarray] = {}
    for name, panel in signals.items():
        sig = _as_panel(panel, str(name))
        if sig.shape != ret.shape:
            raise ValueError(
                f"signal {name!r} shape {sig.shape} != returns {ret.shape}"
            )
        fm = fama_macbeth(sig, ret, lags=1)
        row = {
            "mean_coef": float(fm["mean_coef"]),
            "t_stat": float(fm["t_stat"]),
            "n_dates": int(fm["n_dates"]),
            "gate_date": end,
            "admitted": bool(
                np.isfinite(fm["t_stat"]) and abs(float(fm["t_stat"])) > float(t_min)
            ),
        }
        stats[str(name)] = row
        if row["admitted"]:
            allowlist.append(str(name))
            admitted_panels[str(name)] = sig

    if admitted_panels:
        ortho = orthogonalize_signals(admitted_panels)
        breadth = float(effective_breadth(_signal_corr_matrix(ortho)))
    else:
        breadth = 0.0

    return {
        "allowlist": allowlist,
        "stats": stats,
        "effective_breadth": breadth,
        "selection_end": end,
        "estimand": _ESTIMAND,
        "status": "gated" if allowlist else "empty",
        "t_min": float(t_min),
    }


def run_signal_gate_v2(
    signals: Mapping[str, np.ndarray],
    returns: np.ndarray,
    *,
    dates: Sequence[Any],
    selection_end: str = _DEFAULT_SELECTION_END,
    fdr_q: float = _FDR_Q_DEFAULT,
    hlz_t: float = _HLZ_T_DEFAULT,
    legacy_t_min: float = _LEGACY_T_MIN_DIAGNOSTIC,
) -> dict[str, Any]:
    """Admit signals via BH FDR on Newey-West FM t-stats (PIT fail-closed).

    All-NaN panels are quarantined as ``status=unscored`` (not scored
    rejects). HLZ ``|t_nw| >= hlz_t`` is a discovery disclosure flag only.
    Legacy ``|FM t| > legacy_t_min`` is recorded as a diagnostic, not admission.
    """
    ret = _as_panel(returns, "returns")
    date_list = list(dates)
    if len(date_list) != ret.shape[0]:
        raise ValueError(
            f"dates length {len(date_list)} != returns T={ret.shape[0]}"
        )
    end = _date_str(selection_end)
    parsed = [_date_str(d) for d in date_list]
    bad = [d for d in parsed if d > end]
    if bad:
        raise ValueError(
            f"signal gate refuses dates after selection_end={end}: "
            f"found {bad[0]} (+{len(bad) - 1} more)"
        )

    from src.eval.factor_alpha import bh_fdr

    stats: dict[str, dict[str, Any]] = {}
    scored_names: list[str] = []
    pvalues: list[float] = []
    panels_by_name: dict[str, np.ndarray] = {}

    for name, panel in signals.items():
        sig = _as_panel(panel, str(name))
        if sig.shape != ret.shape:
            raise ValueError(
                f"signal {name!r} shape {sig.shape} != returns {ret.shape}"
            )
        panels_by_name[str(name)] = sig
        if int(np.isfinite(sig).sum()) == 0:
            stats[str(name)] = {
                "mean_coef": float("nan"),
                "t_stat": float("nan"),
                "t_stat_nw": float("nan"),
                "p_value": float("nan"),
                "n_dates": 0,
                "gate_date": end,
                "admitted": False,
                "status": "unscored",
                "discovery_hlz": False,
                "legacy_t_min_pass": False,
            }
            continue
        fm = fama_macbeth_detailed(sig, ret, lags=1)
        t_nw = float(fm["t_stat_nw"])
        t_raw = float(fm["t_stat"])
        n_dates = int(fm["n_dates"])
        p_val = _two_sided_p_from_t(t_nw, n_dates=n_dates)
        row = {
            "mean_coef": float(fm["mean_coef"]),
            "t_stat": t_raw,
            "t_stat_nw": t_nw,
            "p_value": p_val,
            "n_dates": n_dates,
            "gate_date": end,
            "admitted": False,
            "status": "scored",
            "discovery_hlz": bool(
                np.isfinite(t_nw) and abs(t_nw) >= float(hlz_t)
            ),
            "legacy_t_min_pass": bool(
                np.isfinite(t_raw) and abs(t_raw) > float(legacy_t_min)
            ),
        }
        stats[str(name)] = row
        scored_names.append(str(name))
        pvalues.append(p_val if np.isfinite(p_val) else 1.0)

    fdr = bh_fdr(pvalues, q=float(fdr_q)) if scored_names else {
        "ok": True,
        "q": float(fdr_q),
        "m": 0,
        "reject": [],
    }
    reject = list(fdr.get("reject") or [])
    allowlist: list[str] = []
    admitted_panels: dict[str, np.ndarray] = {}
    for i, name in enumerate(scored_names):
        bit = bool(reject[i]) if i < len(reject) else False
        stats[name]["admitted"] = bit
        stats[name]["bh_reject"] = bit
        if bit:
            allowlist.append(name)
            admitted_panels[name] = panels_by_name[name]

    if admitted_panels:
        ortho = orthogonalize_signals(admitted_panels)
        breadth = float(effective_breadth(_signal_corr_matrix(ortho)))
    else:
        breadth = 0.0

    return {
        "allowlist": allowlist,
        "stats": stats,
        "effective_breadth": breadth,
        "selection_end": end,
        "estimand": _ESTIMAND_V2,
        "status": "gated" if allowlist else "empty",
        "fdr_q": float(fdr_q),
        "n_family": int(len(scored_names)),
        "hlz_t": float(hlz_t),
        "legacy_t_min_diagnostic": float(legacy_t_min),
        "t_min": float(legacy_t_min),
    }


def write_signal_allowlist(result: Mapping[str, Any], path: str | Path) -> None:
    """Write gate result to JSON (``config/signal_allowlist.json`` shape).

    Any extra keys on ``result`` beyond the core schema (e.g. ``n_pool``,
    ``pool_secids``, ``factor_names``, ``wall_s`` set by
    ``scripts/run_signal_gate.py``) are passed through as provenance.
    """
    p = Path(path)
    core = {
        "allowlist",
        "status",
        "selection_end",
        "effective_breadth",
        "estimand",
        "stats",
        "t_min",
    }
    payload = {
        "allowlist": list(result.get("allowlist") or []),
        "status": result.get("status", "gated"),
        "selection_end": str(result.get("selection_end", _DEFAULT_SELECTION_END)),
        "effective_breadth": result.get("effective_breadth"),
        "estimand": result.get("estimand", _ESTIMAND),
        "stats": result.get("stats", {}),
        "t_min": result.get("t_min"),
    }
    for k, v in result.items():
        if k not in core:
            payload[k] = v
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_signal_allowlist(path: str | Path) -> dict[str, Any]:
    """Load allowlist JSON from disk."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"allowlist at {p} is not a JSON object")
    return data


def assert_allowlist_valid(
    path: str | Path,
    *,
    selection_end: str = _DEFAULT_SELECTION_END,
) -> dict[str, Any]:
    """Raise if allowlist empty or any admitted gate_date > selection_end."""
    data = load_signal_allowlist(path)
    allow = list(data.get("allowlist") or [])
    if not allow:
        raise ValueError(f"signal allowlist empty (fail-closed): {path}")
    end = _date_str(selection_end)
    stats = data.get("stats") or {}
    for name in allow:
        row = stats.get(name) if isinstance(stats, dict) else None
        if isinstance(row, dict) and "gate_date" in row:
            gd = _date_str(row["gate_date"])
            if gd > end:
                raise ValueError(
                    f"signal {name!r} gate_date={gd} > selection_end={end}"
                )
    file_end = data.get("selection_end")
    if file_end is not None and _date_str(file_end) > end:
        raise ValueError(
            f"allowlist selection_end={file_end} > lock selection_end={end}"
        )
    return data


def load_obs_pack(path: str | Path) -> dict[str, Any]:
    """Load a Lane-B observation pack YAML (geometry / surf_off / cs_admit)."""
    import yaml

    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"obs pack at {p} is not a mapping")
    return dict(raw)


def assert_geometry_pack_valid(path: str | Path) -> dict[str, Any]:
    """Validate a geometry pack for packing into ``iv_surface``.

    ``surf_off`` (empty channels, no resolve) is refused here — use campaign
    lane ``off`` / ``--no-surface-signals`` instead. Packs with
    ``resolve_from: signal_allowlist`` are valid without inline channels.
    """
    from src.data.surface_signals import SURFACE_SIGNAL_NAMES

    data = load_obs_pack(path)
    pack_id = str(data.get("pack_id") or "")
    estimand = str(data.get("estimand") or "")
    if estimand and estimand != _ESTIMAND_GEOMETRY:
        raise ValueError(
            f"obs pack {pack_id!r} estimand={estimand!r} "
            f"!= {_ESTIMAND_GEOMETRY!r}"
        )
    resolve = data.get("resolve_from")
    channels = list(data.get("channels") or [])
    if resolve == "signal_allowlist":
        data["pack_id"] = pack_id or "surf_cs_admit"
        data["estimand"] = _ESTIMAND_GEOMETRY
        data["channels"] = channels
        return data
    if not channels:
        raise ValueError(
            f"geometry pack {pack_id or path!r} has empty channels "
            "(use lane=off for surf_off; refuse empty geometry pack)"
        )
    catalog = set(SURFACE_SIGNAL_NAMES)
    unknown = [c for c in channels if c not in catalog]
    if unknown:
        raise ValueError(
            f"geometry pack {pack_id!r} unknown surface signal(s): {unknown}"
        )
    data["pack_id"] = pack_id
    data["estimand"] = _ESTIMAND_GEOMETRY
    data["channels"] = channels
    return data
