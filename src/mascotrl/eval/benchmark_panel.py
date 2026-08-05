"""Benchmark panel: walk-forward wiring for industry + classic peers.

Applies identical slot mask / optional turnover cap / cost hook to every
registered weight function so CPCV folds compare HAPPO against a fair panel.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.eval.industry_baselines import (
    INDUSTRY_BASELINE_NAMES,
    INDUSTRY_BASELINE_REGISTRY,
    industry_baseline_weights,
)

_EPS = 1e-12
_MIN_OBS = 20

PANEL_EXTRA_NAMES = (
    "cap_weight_bah",
    "low_vol_long",
    "equal_weight_tau_matched",
)

BENCHMARK_PANEL_NAMES: tuple[str, ...] = tuple(INDUSTRY_BASELINE_NAMES) + PANEL_EXTRA_NAMES

WeightFn = Callable[..., np.ndarray]


def list_benchmark_panel() -> tuple[str, ...]:
    return BENCHMARK_PANEL_NAMES


# ---------------------------------------------------------------------------
# Extra panel members
# ---------------------------------------------------------------------------


def _equal_weight(k: int) -> np.ndarray:
    if k <= 0:
        return np.zeros(0, dtype=np.float64)
    return np.full(k, 1.0 / k, dtype=np.float64)


def _renorm_nonneg(w: np.ndarray) -> np.ndarray:
    w = np.nan_to_num(np.asarray(w, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, 0.0)
    s = float(w.sum())
    if s < _EPS:
        return _equal_weight(w.size)
    return w / s


def cap_weight_bah(
    returns_hist: np.ndarray,
    *,
    t: int | None = None,
    w_prev: np.ndarray | None = None,
    mktcap: np.ndarray | None = None,
    mktcap_row: np.ndarray | None = None,
    **_kw: Any,
) -> np.ndarray:
    """Cap-weighted buy-and-hold; equal-weight fallback when mktcap missing.

    ``mktcap`` may be ``(T_full, K)`` with decision index ``t``, or pass a
    single ``mktcap_row`` of shape ``(K,)``. Without either, falls back to
    equal weight (or ``w_prev`` when continuing a BAH path).
    """
    del _kw
    r = np.asarray(returns_hist, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("returns_hist must be (T_hist, K)")
    k = r.shape[1]

    if w_prev is not None:
        prev = np.asarray(w_prev, dtype=np.float64).reshape(-1)
        if prev.size == k and np.all(np.isfinite(prev)) and float(np.sum(np.abs(prev))) > _EPS:
            # Drift buy-and-hold: revalue previous weights by last observed return.
            if r.shape[0] > 0:
                last = np.nan_to_num(r[-1], nan=0.0)
                drifted = prev * (1.0 + last)
                s = float(np.sum(np.abs(drifted)))
                if s > _EPS:
                    return drifted / s
            return prev.copy()

    row = None
    if mktcap_row is not None:
        row = np.asarray(mktcap_row, dtype=np.float64).reshape(-1)
    elif mktcap is not None and t is not None:
        mc = np.asarray(mktcap, dtype=np.float64)
        if mc.ndim == 2 and mc.shape[1] == k and mc.shape[0] > 0:
            # PIT: last available cap strictly before decision t (or row 0 at t=0).
            idx = max(0, min(int(t) - 1, mc.shape[0] - 1)) if int(t) > 0 else 0
            row = mc[idx]
        elif mc.ndim == 1 and mc.size == k:
            row = mc

    if row is None or row.size != k or not np.any(np.isfinite(row) & (row > 0)):
        return _equal_weight(k)
    row = np.where(np.isfinite(row) & (row > 0), row, 0.0)
    return _renorm_nonneg(row)


def low_vol_long(
    returns_hist: np.ndarray,
    *,
    t: int | None = None,
    w_prev: np.ndarray | None = None,
    lookback: int = 63,
    min_obs: int = _MIN_OBS,
    **_kw: Any,
) -> np.ndarray:
    """Long-only equal weight in bottom quintile of trailing HV."""
    del t, w_prev, _kw
    r = np.asarray(returns_hist, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("returns_hist must be (T_hist, K)")
    t_hist, k = r.shape
    if k == 0:
        return np.zeros(0, dtype=np.float64)
    if t_hist < int(min_obs):
        return _equal_weight(k)

    window = r[-int(lookback) :] if t_hist >= int(lookback) else r
    hv = np.full(k, np.nan, dtype=np.float64)
    for j in range(k):
        col = window[:, j]
        finite = col[np.isfinite(col)]
        if finite.size >= int(min_obs):
            v = float(np.std(finite, ddof=1)) if finite.size > 1 else float(np.std(finite))
            hv[j] = v if np.isfinite(v) else np.nan

    finite_idx = np.where(np.isfinite(hv))[0]
    if finite_idx.size == 0:
        return _equal_weight(k)

    # Bottom quintile among finite names (at least 1).
    n_pick = max(1, int(np.ceil(finite_idx.size / 5.0)))
    order = finite_idx[np.argsort(hv[finite_idx])]
    picks = order[:n_pick]
    w = np.zeros(k, dtype=np.float64)
    w[picks] = 1.0 / picks.size
    return w


def equal_weight_tau_matched(
    returns_hist: np.ndarray,
    *,
    t: int | None = None,
    w_prev: np.ndarray | None = None,
    **_kw: Any,
) -> np.ndarray:
    """Equal-weight target; hard turnover projection applied by the scorer."""
    del t, w_prev, _kw
    r = np.asarray(returns_hist, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("returns_hist must be (T_hist, K)")
    return _equal_weight(int(r.shape[1]))


PANEL_EXTRA_REGISTRY: dict[str, WeightFn] = {
    "cap_weight_bah": cap_weight_bah,
    "low_vol_long": low_vol_long,
    "equal_weight_tau_matched": equal_weight_tau_matched,
}


def get_weight_fn(name: str) -> WeightFn:
    """Resolve a panel member name to a weight callable."""
    if name in INDUSTRY_BASELINE_REGISTRY:

        def _industry(
            returns_hist: np.ndarray,
            *,
            t: int | None = None,
            w_prev: np.ndarray | None = None,
            **kw: Any,
        ) -> np.ndarray:
            del kw
            tt = 0 if t is None else int(t)
            return industry_baseline_weights(
                name, returns_hist=returns_hist, t=tt, w_prev=w_prev
            )

        return _industry
    if name in PANEL_EXTRA_REGISTRY:
        return PANEL_EXTRA_REGISTRY[name]
    raise KeyError(f"unknown benchmark panel member: {name!r}")


# ---------------------------------------------------------------------------
# Walk-forward runner
# ---------------------------------------------------------------------------


def _apply_slot_mask(w: np.ndarray, mask_row: np.ndarray | None) -> np.ndarray:
    if mask_row is None:
        return w
    m = np.asarray(mask_row, dtype=bool).reshape(-1)
    out = np.asarray(w, dtype=np.float64).copy()
    if m.size != out.size:
        raise ValueError(f"slot_mask row size {m.size} != weights {out.size}")
    out = out * m.astype(np.float64)
    return out


def _clip_turnover(w: np.ndarray, w_prev: np.ndarray, cap: float) -> np.ndarray:
    dw = w - w_prev
    turn = float(np.sum(np.abs(dw)))
    if not np.isfinite(turn) or turn <= float(cap) or turn < _EPS:
        return w
    scale = float(cap) / turn
    return w_prev + dw * scale


def _default_gross_pnl(w: np.ndarray, w_prev: np.ndarray, ret: np.ndarray) -> float:
    del w_prev
    r = np.nan_to_num(np.asarray(ret, dtype=np.float64), nan=0.0)
    return float(np.nansum(np.asarray(w, dtype=np.float64) * r))


def _make_friction_cost_fn(
    friction_kwargs: Mapping[str, Any] | None,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float] | None:
    if not friction_kwargs:
        return None
    import torch

    from src.eval.friction import apply_costs

    kwargs = dict(friction_kwargs)

    def _fn(w: np.ndarray, w_prev: np.ndarray, ret: np.ndarray) -> float:
        wt = torch.as_tensor(w, dtype=torch.float32)
        wp = torch.as_tensor(w_prev, dtype=torch.float32)
        rt = torch.as_tensor(ret, dtype=torch.float32)
        fb = apply_costs(wt, wp, rt, **kwargs)
        return float(fb.net)

    return _fn


def run_benchmark_on_fold(
    name: str,
    returns: np.ndarray,
    *,
    w_fn: WeightFn | None = None,
    slot_mask: np.ndarray | None = None,
    apply_costs_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float] | None = None,
    friction_kwargs: Mapping[str, Any] | None = None,
    turnover_cap: float | None = None,
    mktcap: np.ndarray | None = None,
    return_meta: bool = False,
    min_hist: int = 0,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Walk day-by-day; weights at ``t`` use only ``returns[:t]`` (no lookahead).

    Parameters
    ----------
    name
        Registry key (or free label when ``w_fn`` is supplied).
    returns
        Shape ``(T, K)`` simple returns.
    w_fn
        Optional override; signature ``(returns_hist, *, t, w_prev, ...)``.
    slot_mask
        Optional ``(T, K)`` bool; False zeros that slot before PnL.
    apply_costs_fn
        ``(w, w_prev, ret_t) -> net_pnl``. When omitted, uses gross ``w·r``
        unless ``friction_kwargs`` is set (then ``src.eval.friction.apply_costs``).
    turnover_cap
        Optional L1 ``|Δw|`` clip vs previous weights.
    mktcap
        Optional ``(T, K)`` market-cap panel for ``cap_weight_bah``.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("returns must be (T, K)")
    t_len, k = r.shape

    mask = None
    if slot_mask is not None:
        mask = np.asarray(slot_mask, dtype=bool)
        if mask.shape != r.shape:
            raise ValueError(f"slot_mask shape {mask.shape} != returns {r.shape}")

    fn = w_fn if w_fn is not None else get_weight_fn(name)
    cost_fn = apply_costs_fn
    if cost_fn is None:
        cost_fn = _make_friction_cost_fn(friction_kwargs)
    if cost_fn is None:
        cost_fn = _default_gross_pnl

    pnl = np.zeros(t_len, dtype=np.float64)
    weights = np.zeros((t_len, k), dtype=np.float64)
    w_prev = np.zeros(k, dtype=np.float64)

    for t in range(t_len):
        hist = r[:t]
        if t < int(min_hist):
            w = np.zeros(k, dtype=np.float64)
        else:
            kw: dict[str, Any] = {"t": t, "w_prev": w_prev}
            if mktcap is not None:
                kw["mktcap"] = mktcap
            try:
                w = np.asarray(
                    fn(hist, **kw), dtype=np.float64
                ).reshape(-1)
            except TypeError:
                try:
                    w = np.asarray(
                        fn(returns_hist=hist, **kw), dtype=np.float64
                    ).reshape(-1)
                except TypeError:
                    # Pure ``(returns_hist,) -> w`` callables (e.g. OLPS).
                    w = np.asarray(fn(hist), dtype=np.float64).reshape(-1)
            if w.size != k:
                raise ValueError(f"{name}: weight size {w.size} != K={k}")
            w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

        if mask is not None:
            w = _apply_slot_mask(w, mask[t])

        if turnover_cap is not None and turnover_cap > 0:
            w = _clip_turnover(w, w_prev, float(turnover_cap))
            if mask is not None:
                w = _apply_slot_mask(w, mask[t])

        ret_t = r[t]
        if mask is not None:
            ret_t = np.where(mask[t], np.nan_to_num(ret_t, nan=0.0), 0.0)
        else:
            ret_t = np.nan_to_num(ret_t, nan=0.0)

        step_pnl = float(cost_fn(w, w_prev, ret_t))
        if not np.isfinite(step_pnl):
            step_pnl = 0.0
        pnl[t] = step_pnl
        weights[t] = w
        w_prev = w.copy()

    if return_meta:
        return pnl, {"weights": weights, "name": name}
    return pnl


def run_panel(
    names: Sequence[str],
    returns: np.ndarray,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """Run multiple panel members; returns ``{name: pnl_series}``."""
    out: dict[str, np.ndarray] = {}
    # Strip return_meta so callers always get series.
    kwargs = dict(kwargs)
    kwargs.pop("return_meta", None)
    for name in names:
        series = run_benchmark_on_fold(name, returns, return_meta=False, **kwargs)
        assert isinstance(series, np.ndarray)
        out[str(name)] = series
    return out


def attach_fold_benchmark_panel(
    fold: Any,
    *,
    dates: Sequence[Any],
    returns: np.ndarray,
    names: Sequence[str] | None = None,
    friction_kwargs: Mapping[str, Any] | None = None,
    turnover_cap: float | None = None,
    slot_mask: np.ndarray | None = None,
    mktcap: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run fold-symmetric ``run_panel`` on test-window rows; attach serializable keys.

    ``returns`` must be ``(T, K)`` aligned with ``dates``. Default ``names`` is the
    full ``BENCHMARK_PANEL_NAMES`` registry; pass a short list in unit tests.
    """
    import pandas as pd

    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("returns must be (T, K)")
    date_list = list(dates)
    if len(date_list) != r.shape[0]:
        raise ValueError(
            f"dates length {len(date_list)} != returns rows {r.shape[0]}"
        )
    idx = {
        str(pd.Timestamp(d).date()): i for i, d in enumerate(date_list)
    }
    test_idx: list[int] = []
    for win in getattr(fold, "test_windows", None) or []:
        lo = idx.get(str(win.get("start")))
        hi = idx.get(str(win.get("end")))
        if lo is None or hi is None:
            continue
        test_idx.extend(range(int(lo), int(hi) + 1))
    if not test_idx:
        return {
            "ok": False,
            "reason": "no_test_indices",
            "fold_id": int(getattr(fold, "fold_id", -1)),
            "panel_keys": [],
        }
    # Preserve chronological order, drop duplicates from multi-window folds.
    seen: set[int] = set()
    ordered: list[int] = []
    for i in test_idx:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    sub = r[ordered]
    mask_sub = None
    if slot_mask is not None:
        mask_sub = np.asarray(slot_mask, dtype=bool)[ordered]
    mkt_sub = None
    if mktcap is not None:
        mkt_sub = np.asarray(mktcap, dtype=np.float64)[ordered]
    panel_names = list(names) if names is not None else list(BENCHMARK_PANEL_NAMES)
    series = run_panel(
        panel_names,
        sub,
        friction_kwargs=friction_kwargs,
        turnover_cap=turnover_cap,
        slot_mask=mask_sub,
        mktcap=mkt_sub,
    )
    pnl_lists = {k: v.tolist() for k, v in series.items()}
    return {
        "ok": True,
        "fold_id": int(getattr(fold, "fold_id", -1)),
        "n_days": int(sub.shape[0]),
        "panel_keys": sorted(pnl_lists.keys()),
        "pnls": pnl_lists,
        "protocol": "fold_symmetric_benchmark_panel",
    }
