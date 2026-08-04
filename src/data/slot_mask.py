"""Time-varying universe under a fixed slot count K (Phase E).

The policy has a fixed asset dimension, so a naively time-varying universe is
architecturally inadmissible. Slot masking keeps K slots, assigns PIT-eligible
names into those slots at each rebalance, and carries a validity mask that
zeroes both weight and label contribution for inactive slots.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def select_slots_for_date(
    eligible_secids: Sequence[int],
    k: int = 50,
) -> list[int | None]:
    """
    Fill ``k`` slots from PIT-eligible security IDs for one rebalance date.

    Takes the first ``k`` IDs in the given order; pads with ``None`` when fewer
    than ``k`` names are eligible. Excess names beyond ``k`` are dropped (caller
    is responsible for ranking / screening before this call).
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    slots: list[int | None] = [None] * k
    for i, sid in enumerate(eligible_secids):
        if i >= k:
            break
        slots[i] = int(sid)
    return slots


def valid_mask_from_slots(slots: Sequence[int | None]) -> np.ndarray:
    """Boolean mask of length K: True where a slot holds an active secid."""
    return np.asarray([s is not None for s in slots], dtype=bool)


def apply_slot_mask(w: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Zero inactive slots in a weight vector (or last axis of a weight array).

    ``w`` and ``mask`` must broadcast: ``mask`` is length-K along the last axis
    of ``w``.
    """
    w_arr = np.asarray(w, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    if m.ndim != 1:
        raise ValueError("mask must be 1-D")
    if w_arr.shape[-1] != m.shape[0]:
        raise ValueError(
            f"weight last dim {w_arr.shape[-1]} != mask length {m.shape[0]}"
        )
    return w_arr * m


# @lat: [[invariants#Slot mask]]
def coverage_masks_from_features(
    atm: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Build a ``(T, K)`` validity mask from feature / label coverage.

    A slot is active only when both ATM IV and the realized label are finite
    on that day. Callers that supply an explicit rebalance schedule should
    intersect this with membership masks.
    """
    a = np.asarray(atm)
    lab = np.asarray(labels)
    if a.shape != lab.shape:
        raise ValueError(f"atm shape {a.shape} != labels shape {lab.shape}")
    return np.isfinite(a) & np.isfinite(lab)


def _equity_coverage(
    equity_labels: np.ndarray,
    spot: np.ndarray | None = None,
) -> np.ndarray:
    """Equity slot active when label is finite; optional spot finiteness gate."""
    lab = np.asarray(equity_labels)
    mask = np.isfinite(lab)
    if spot is not None:
        sp = np.asarray(spot)
        if sp.shape != lab.shape:
            raise ValueError(f"spot shape {sp.shape} != equity_labels {lab.shape}")
        mask = mask & np.isfinite(sp)
    return mask


def coverage_masks_for_arm(
    *,
    arm,
    atm: np.ndarray | None = None,
    option_labels: np.ndarray | None = None,
    equity_labels: np.ndarray | None = None,
    spot: np.ndarray | None = None,
    label_stem_opt: str | None = None,
    label_stem_eq: str | None = None,
    panel=None,
    dates_idx=None,
) -> np.ndarray:
    """
    Block-specific coverage masks for spectrum arms. Shape ``(T, n_slots)``.

    - Option slots: ATM IV and option label both finite (status quo).
    - Equity slots: equity label finite; ATM IV is **not** required. When
      ``spot`` is provided, require spot finite as well.
    - Mix: concatenate option then equity blocks.

    Array inputs are preferred for unit tests and eval wiring. Optional
    ``panel`` / ``dates_idx`` / stem kwargs resolve matrices via
    ``wide_feature_matrix`` when arrays are omitted.
    """
    from src.arms.spec import ArmSpec

    if not isinstance(arm, ArmSpec):
        raise TypeError(f"arm must be ArmSpec, got {type(arm)!r}")

    def _from_panel(stem: str, n: int) -> np.ndarray:
        if panel is None:
            raise ValueError(f"panel required to resolve stem={stem!r}")
        from src.data.oos_panel import wide_feature_matrix

        mat = wide_feature_matrix(panel, stem, n)
        if dates_idx is not None:
            mat = np.asarray(mat)[np.asarray(dates_idx)]
        return mat

    opt_n = int(arm.option_slots)
    eq_n = int(arm.equity_slots)
    blocks: list[np.ndarray] = []

    if opt_n:
        if option_labels is None:
            stem = label_stem_opt or arm.option_label_stem
            option_labels = _from_panel(stem, opt_n)
        if atm is None:
            atm = _from_panel("atm_iv", opt_n)
        a = np.asarray(atm)
        ol = np.asarray(option_labels)
        # Allow a wider ATM panel; take the option block columns.
        if a.shape[-1] > opt_n:
            a = a[..., :opt_n]
        if ol.shape[-1] > opt_n:
            ol = ol[..., :opt_n]
        blocks.append(coverage_masks_from_features(a, ol))

    if eq_n:
        if equity_labels is None:
            stem = label_stem_eq or arm.equity_label_stem
            equity_labels = _from_panel(stem, eq_n)
        el = np.asarray(equity_labels)
        if el.shape[-1] > eq_n:
            el = el[..., :eq_n]
        sp = None
        if spot is not None:
            sp = np.asarray(spot)
            if sp.shape[-1] > eq_n:
                sp = sp[..., :eq_n]
        elif panel is not None:
            try:
                sp = _from_panel("spot", eq_n)
            except (KeyError, ValueError):
                sp = None
        blocks.append(_equity_coverage(el, sp))

    if not blocks:
        raise ValueError("arm has no slots")
    if len(blocks) == 1:
        return blocks[0]
    if blocks[0].shape[0] != blocks[1].shape[0]:
        raise ValueError(
            f"option/equity T mismatch {blocks[0].shape[0]} vs {blocks[1].shape[0]}"
        )
    return np.concatenate(blocks, axis=-1)


def membership_masks_for_fixed_slots(
    dates: Sequence,
    secids: Sequence[int],
    *,
    tickers: Sequence[str],
    members_by_date: dict,
    coverage: np.ndarray | None = None,
) -> np.ndarray:
    """
    PIT membership masks for a fixed slot layout (panel columns = secids).

    Slot ``i`` is active on date ``d`` when ``tickers[i]`` is in the PIT
    membership set for ``d``. Optionally intersect with a coverage mask so
    inactive / missing marks stay zero. This is true membership gating on the
    frozen K-slot panel; it is not a full re-ranking of the entire lake into
    new slot assignments (that would require rematerializing features).
    """
    if len(secids) != len(tickers):
        raise ValueError("secids and tickers must have equal length")
    k = len(secids)
    tick_u = [str(t).strip().upper() for t in tickers]
    masks = np.zeros((len(dates), k), dtype=bool)
    for i, d in enumerate(dates):
        members = _lookup_eligible(members_by_date, d)
        member_set = {
            str(x).strip().upper() for x in (members or []) if x is not None
        }
        for j, tic in enumerate(tick_u):
            masks[i, j] = tic in member_set if member_set else False
        # If membership snapshot is empty for the date, fall back to all-on
        # so we do not silently wipe the panel; caller should disclose.
        if not member_set:
            masks[i, :] = True
    if coverage is not None:
        cov = np.asarray(coverage, dtype=bool)
        if cov.shape != masks.shape:
            raise ValueError(
                f"coverage shape {cov.shape} != membership masks {masks.shape}"
            )
        masks = masks & cov
    return masks


def build_members_by_date_from_intervals(
    dates: Sequence,
    membership_df,
    *,
    ticker_col: str = "ticker",
    start_col: str = "start_date",
    end_col: str = "end_date",
) -> dict:
    """
    Expand interval membership rows into ``{date: [tickers...]}`` for the
    evaluation calendar. Empty when the snapshot is missing.

    Uses a single sort + two-pointer sweep so cost is O((T + N) log N) rather
    than O(T · N) pandas filters per date (which made hist OOS unusable).
    """
    import pandas as pd

    if membership_df is None or len(membership_df) == 0:
        return {}
    df = membership_df.copy()
    df[ticker_col] = df[ticker_col].astype(str).str.strip().str.upper()
    df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
    df[end_col] = pd.to_datetime(df[end_col], errors="coerce")
    df = df[df[start_col].notna()].sort_values(start_col)
    starts = df[start_col].to_numpy()
    ends = df[end_col].to_numpy()
    tickers = df[ticker_col].to_list()
    # Sentinel far-future for open-ended memberships.
    far = np.datetime64("2262-04-11")
    end_ord = np.array(
        [far if pd.isna(e) else np.datetime64(e) for e in ends],
        dtype="datetime64[ns]",
    )
    start_ord = starts.astype("datetime64[ns]")

    date_ts = [pd.Timestamp(d).normalize() for d in dates]
    order = np.argsort(np.array(date_ts, dtype="datetime64[ns]"))
    out: dict = {}
    active: dict[str, int] = {}
    i_start = 0
    n = len(tickers)
    # Process dates in chronological order; add/remove interval endpoints.
    # Removals: track end events in a min-heap of (end, ticker).
    import heapq

    end_heap: list[tuple[np.datetime64, str]] = []
    for oi in order:
        ts = date_ts[oi]
        t64 = np.datetime64(ts)
        while i_start < n and start_ord[i_start] <= t64:
            tic = tickers[i_start]
            active[tic] = active.get(tic, 0) + 1
            heapq.heappush(end_heap, (end_ord[i_start], tic))
            i_start += 1
        while end_heap and end_heap[0][0] < t64:
            _, tic = heapq.heappop(end_heap)
            cnt = active.get(tic, 0) - 1
            if cnt <= 0:
                active.pop(tic, None)
            else:
                active[tic] = cnt
        members = list(active.keys())
        out[ts] = members
        out[str(ts.date())] = members
    return out



def build_slot_masks_over_dates(
    dates: Sequence,
    eligible_by_date: dict,
    *,
    k: int = 50,
) -> tuple[list[list[int | None]], np.ndarray]:
    """
    Assign slots independently at each date from a PIT-eligible map.

    ``eligible_by_date`` maps a date-like key to an ordered sequence of secids.
    Missing dates yield an all-inactive mask. Truncating the date list or
    mutating future eligibility must not change past slot assignments (tested).
    """
    slots_rows: list[list[int | None]] = []
    masks = np.zeros((len(dates), int(k)), dtype=bool)
    for i, d in enumerate(dates):
        eligible = _lookup_eligible(eligible_by_date, d)
        slots = select_slots_for_date(list(eligible or []), k=k)
        slots_rows.append(slots)
        masks[i] = valid_mask_from_slots(slots)
    return slots_rows, masks


def _lookup_eligible(eligible_by_date: dict, d) -> Sequence[int] | None:
    if d in eligible_by_date:
        return eligible_by_date[d]
    try:
        import pandas as pd

        ts = pd.Timestamp(d)
        for key in (ts, ts.normalize(), ts.date(), str(ts.date())):
            if key in eligible_by_date:
                return eligible_by_date[key]
    except Exception:
        pass
    return None


def masked_pnl(
    w: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Portfolio P&L with inactive slots zeroed in both weights and labels.

    For 1-D ``(K,)`` inputs returns a scalar; for ``(T, K)`` returns shape
    ``(T,)`` (sum over names per day).
    """
    w_arr = np.asarray(w, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    if w_arr.shape != lab.shape:
        raise ValueError(f"w shape {w_arr.shape} != labels shape {lab.shape}")
    if m.ndim != 1 or w_arr.shape[-1] != m.shape[0]:
        raise ValueError(
            f"mask length {m.shape} incompatible with weight shape {w_arr.shape}"
        )
    contrib = np.nan_to_num(w_arr * lab * m, nan=0.0)
    return contrib.sum(axis=-1)
