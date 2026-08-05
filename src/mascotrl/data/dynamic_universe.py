"""Time-varying (dynamic) universe selection under a fixed slot count K (W4.1).

Complements ``src.data.slot_mask`` (which assigns already-eligible names into
K slots) with the eligibility screening and per-rebalance re-selection loop
that produce those PIT-eligible names in the first place, plus HRP/liquidity
control arms for the bakeoff until CRUCIBLE lands.

Every date-indexed lookup here is trailing-window only: a rebalance at index
``t`` never reads a row after ``t`` (eligibility) or after ``t-1`` (return
history used to select names), so re-selection cannot see the future.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.slot_mask import select_slots_for_date, valid_mask_from_slots


def _asof_index(dates: Sequence, asof: Any) -> int:
    """Row index of the last date <= ``asof`` (PIT: never a future row)."""
    ts = pd.Timestamp(asof)
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    pos = idx.searchsorted(ts, side="right") - 1
    if pos < 0:
        raise ValueError(f"asof {asof} is before the first available date {dates[0]}")
    return int(pos)


def option_eligibility_screen(
    *,
    asof: Any,
    secids: Sequence[int],
    signal_panels: Mapping[str, np.ndarray],
    dates: Sequence,
    trailing_days: int = 63,
    min_obs: int = 21,
    required_signals: Sequence[str] = ("mfis_30", "mfis_365"),
    volume_panel: np.ndarray | None = None,
    min_option_volume: float = 0.0,
    max_missing_frac: float = 1.0,
) -> list[int]:
    """PIT eligibility: require enough trailing observations of every signal.

    For each ``secid``, looks at the trailing window ending at ``asof``
    (inclusive of ``asof``, never a future row). A secid is eligible only
    when every column in ``required_signals`` has at least ``min_obs``
    finite observations in that window.

    Optional liquidity gate: when ``volume_panel`` is provided and
    ``min_option_volume > 0``, require mean finite volume in the same
    trailing window to meet the floor. ``max_missing_frac`` caps the
    fraction of NaN volume rows allowed in that window.
    """
    for name in required_signals:
        if name not in signal_panels:
            raise ValueError(f"required signal {name!r} missing from signal_panels")

    end = _asof_index(dates, asof)
    start = max(0, end - int(trailing_days) + 1)
    secids_list = list(secids)
    vol = None if volume_panel is None else np.asarray(volume_panel, dtype=float)

    eligible: list[int] = []
    for j, sid in enumerate(secids_list):
        ok = True
        for name in required_signals:
            panel = np.asarray(signal_panels[name])
            if panel.ndim != 2 or j >= panel.shape[1]:
                ok = False
                break
            window = panel[start : end + 1, j]
            if int(np.isfinite(window).sum()) < int(min_obs):
                ok = False
                break
        if ok and vol is not None and float(min_option_volume) > 0.0:
            if vol.ndim != 2 or j >= vol.shape[1]:
                ok = False
            else:
                vwin = vol[start : end + 1, j]
                miss = float(np.mean(~np.isfinite(vwin))) if vwin.size else 1.0
                if miss > float(max_missing_frac):
                    ok = False
                else:
                    finite = vwin[np.isfinite(vwin)]
                    mean_vol = float(np.mean(finite)) if finite.size else 0.0
                    if mean_vol < float(min_option_volume):
                        ok = False
        if ok:
            eligible.append(int(sid))
    return eligible


def _lookup_eligible_by_date(eligibility_by_date: dict | None, d) -> Sequence[int] | None:
    if eligibility_by_date is None:
        return None
    if d in eligibility_by_date:
        return eligibility_by_date[d]
    ts = pd.Timestamp(d)
    for key in (ts, ts.normalize(), ts.date(), str(ts.date())):
        if key in eligibility_by_date:
            return eligibility_by_date[key]
    return None


def build_dynamic_universe(
    *,
    dates: Sequence,
    rebalance_mask: np.ndarray,
    wide_returns: np.ndarray,
    secids: Sequence[int],
    k: int,
    select_fn: Callable[..., dict],
    trailing_days: int = 252,
    select_kwargs: dict | None = None,
    eligibility_by_date: dict | None = None,
) -> tuple[list[list[int | None]], np.ndarray, list[dict]]:
    """Re-select ``k`` slots on every rebalance day from a trailing PIT window.

    On each rebalance day ``t``, restricts columns to the PIT-eligible
    secids for that date (all secids when ``eligibility_by_date`` is
    ``None``), takes the trailing return window ending at ``t-1`` (PIT: the
    return realized on the rebalance day itself is not used to pick that
    day's universe), calls ``select_fn`` on that window, and maps the result
    onto ``k`` fixed slots via ``select_slots_for_date``. Non-rebalance days
    hold the previous slot assignment.
    """
    mask = np.asarray(rebalance_mask, dtype=bool).reshape(-1)
    rets = np.asarray(wide_returns, dtype=np.float64)
    t_n = len(dates)
    if mask.size != t_n:
        raise ValueError(f"rebalance_mask length {mask.size} != len(dates)={t_n}")
    if rets.shape[0] != t_n:
        raise ValueError(f"wide_returns rows {rets.shape[0]} != len(dates)={t_n}")
    secids_list = list(secids)
    if rets.shape[1] != len(secids_list):
        raise ValueError(f"wide_returns cols {rets.shape[1]} != len(secids)={len(secids_list)}")
    kwargs = dict(select_kwargs or {})

    slots_rows: list[list[int | None]] = []
    selection_log: list[dict] = []
    current: list[int | None] = [None] * int(k)

    for t in range(t_n):
        if mask[t]:
            end_idx = t - 1 if t >= 1 else t
            start_idx = max(0, end_idx - int(trailing_days) + 1)
            window = rets[start_idx : end_idx + 1]

            eligible = _lookup_eligible_by_date(eligibility_by_date, dates[t])
            if eligible is None:
                elig_cols = list(range(len(secids_list)))
            else:
                elig_set = {int(s) for s in eligible}
                elig_cols = [i for i, s in enumerate(secids_list) if int(s) in elig_set]

            elig_secids = [secids_list[i] for i in elig_cols]
            elig_window = window[:, elig_cols] if elig_cols else window[:, :0]
            call_kwargs = dict(kwargs)
            state_blocks = call_kwargs.get("state_blocks")
            if state_blocks is not None:
                # state_blocks is (T_full, D_pool[, B]); restrict it to the
                # same trailing window + eligible columns as elig_window so
                # select_fn's (returns, state_blocks) shapes stay aligned
                # even when eligibility drops pool members on this date.
                sb_window = np.asarray(state_blocks)[start_idx : end_idx + 1]
                call_kwargs["state_blocks"] = (
                    sb_window[:, elig_cols] if elig_cols else sb_window[:, :0]
                )
            k_eff = min(int(k), len(elig_cols))
            if k_eff > 0:
                call_kwargs.pop("_dii_cache", None)
                result = select_fn(elig_window, k=k_eff, secids=elig_secids, **call_kwargs)
                out_secids = result.get("secids")
                if out_secids is None:
                    idx = result.get("indices") or []
                    out_secids = [elig_secids[i] for i in idx]
            else:
                out_secids = []

            current = select_slots_for_date(out_secids, k=int(k))
            selection_log.append(
                {
                    "date": dates[t],
                    "secids": [s for s in current if s is not None],
                    "n_eligible": len(elig_cols),
                }
            )

        slots_rows.append(list(current))

    valid_mask = np.array([valid_mask_from_slots(row) for row in slots_rows], dtype=bool)
    return slots_rows, valid_mask, selection_log


def build_slotted_panel(
    *,
    dates: Sequence,
    slots_rows: Sequence[Sequence[int | None]],
    wide_returns: np.ndarray,
    col_map: Mapping[int, int],
) -> np.ndarray:
    """Map each slot's occupant secid to its realized return at each date.

    Column ``j`` at row ``t`` is the return of whichever secid occupies
    slot ``j`` on that date, or ``0.0`` when the slot is inactive (``None``
    or a secid absent from ``col_map``); the paired ``valid_mask`` from
    ``build_dynamic_universe`` is what actually zeros an inactive slot's
    weight / label contribution downstream, so 0.0 here is a safe filler
    that keeps the returns panel finite.
    """
    rets = np.asarray(wide_returns, dtype=np.float64)
    t_n = len(dates)
    if len(slots_rows) != t_n:
        raise ValueError(f"slots_rows length {len(slots_rows)} != len(dates)={t_n}")
    k = len(slots_rows[0]) if slots_rows else 0
    out = np.zeros((t_n, k), dtype=np.float64)
    for t, row in enumerate(slots_rows):
        if len(row) != k:
            raise ValueError(f"slots_rows[{t}] length {len(row)} != k={k}")
        for j, sid in enumerate(row):
            if sid is None:
                continue
            col = col_map.get(int(sid))
            if col is None:
                continue
            out[t, j] = rets[t, col]
    return out


def selection_turnover(slots_rows: Sequence[Sequence[int | None]]) -> dict[str, Any]:
    """Names added / dropped at each *distinct* rebalance (hold rows skipped)."""
    distinct: list[list[int | None]] = []
    for row in slots_rows:
        row_l = list(row)
        if not distinct or row_l != distinct[-1]:
            distinct.append(row_l)

    per_step: list[dict] = []
    for prev, curr in zip(distinct[:-1], distinct[1:]):
        prev_set = {s for s in prev if s is not None}
        curr_set = {s for s in curr if s is not None}
        added = sorted(curr_set - prev_set)
        dropped = sorted(prev_set - curr_set)
        per_step.append(
            {"added": added, "dropped": dropped, "n_added": len(added), "n_dropped": len(dropped)}
        )

    mean_added = float(np.mean([p["n_added"] for p in per_step])) if per_step else 0.0
    mean_dropped = float(np.mean([p["n_dropped"] for p in per_step])) if per_step else 0.0
    return {"mean_added": mean_added, "mean_dropped": mean_dropped, "per_step": per_step}


def _greedy_farthest_point(dist: np.ndarray, k: int, d: int) -> list[int]:
    if d == 0:
        return []
    seed = int(np.argmax(dist.sum(axis=1)))
    chosen = [seed]
    while len(chosen) < k:
        remaining = [i for i in range(d) if i not in chosen]
        if not remaining:
            break
        nxt = max(remaining, key=lambda i: min(float(dist[i, c]) for c in chosen))
        chosen.append(nxt)
    return chosen


def select_universe_corr_cluster(
    returns: np.ndarray, *, k: int, secids: Sequence[int] | None = None, **_kw: Any
) -> dict[str, Any]:
    """HRP-like control: pick ``k`` diverse names by |correlation| clustering.

    Uses scipy hierarchical clustering (average linkage on ``1 - |corr|``
    distance) into ``k`` clusters, taking the most central member of each
    cluster. Falls back to greedy farthest-point selection on the same
    distance matrix when scipy is unavailable.
    """
    x = np.asarray(returns, dtype=np.float64)
    d = int(x.shape[1]) if x.ndim == 2 else 0
    k_eff = max(0, min(int(k), d))
    if k_eff == 0:
        return {"indices": [], "secids": [] if secids is not None else None, "provenance": "corr_cluster"}
    if d == 1:
        selected = [0]
    else:
        corr = np.corrcoef(np.nan_to_num(x, nan=0.0), rowvar=False)
        corr = np.nan_to_num(np.atleast_2d(corr), nan=0.0)
        dist = 1.0 - np.abs(corr)
        dist = (dist + dist.T) / 2.0
        np.fill_diagonal(dist, 0.0)
        try:
            from scipy.cluster.hierarchy import fcluster, linkage
            from scipy.spatial.distance import squareform

            condensed = squareform(dist, checks=False)
            z = linkage(condensed, method="average")
            clusters = fcluster(z, t=k_eff, criterion="maxclust")
            selected = []
            for c in sorted(set(clusters.tolist())):
                members = [i for i in range(d) if clusters[i] == c]
                if len(members) == 1:
                    selected.append(members[0])
                else:
                    avg_dist = dist[np.ix_(members, members)].mean(axis=1)
                    selected.append(members[int(np.argmin(avg_dist))])
            selected = sorted(set(selected))
            provenance = "corr_cluster"
            if len(selected) < k_eff:
                # Degenerate clustering (e.g. ties collapsed clusters);
                # top up with farthest-point picks from the remaining pool.
                remaining_pool = [i for i in range(d) if i not in selected]
                filler = _greedy_farthest_point(dist[np.ix_(remaining_pool, remaining_pool)], k_eff - len(selected), len(remaining_pool))
                selected.extend(remaining_pool[i] for i in filler)
        except ImportError:
            selected = _greedy_farthest_point(dist, k_eff, d)
            provenance = "corr_cluster_greedy_fallback"
    selected = sorted(selected)[:k_eff]
    secid_out = [int(secids[i]) for i in selected] if secids is not None else None
    return {"indices": selected, "secids": secid_out, "provenance": provenance}


def select_universe_liquidity(
    returns: np.ndarray, *, k: int, secids: Sequence[int] | None = None, **_kw: Any
) -> dict[str, Any]:
    """Rank by non-NaN coverage, then by ``1/vol`` (liquidity proxy sans ADV)."""
    x = np.asarray(returns, dtype=np.float64)
    d = int(x.shape[1]) if x.ndim == 2 else 0
    k_eff = max(0, min(int(k), d))
    if k_eff == 0:
        return {"indices": [], "secids": [] if secids is not None else None, "provenance": "liquidity"}
    coverage = np.isfinite(x).mean(axis=0)
    vol = np.nanstd(x, axis=0)
    vol = np.where(np.isfinite(vol) & (vol > 0), vol, np.inf)
    inv_vol = np.where(np.isfinite(vol), 1.0 / vol, 0.0)
    order = sorted(range(d), key=lambda i: (-coverage[i], -inv_vol[i]))
    selected = sorted(order[:k_eff])
    secid_out = [int(secids[i]) for i in selected] if secids is not None else None
    return {"indices": selected, "secids": secid_out, "provenance": "liquidity"}
