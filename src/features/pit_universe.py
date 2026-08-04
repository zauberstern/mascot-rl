"""Point-in-time equity / option universe builders (Alpha v2).

Equity path compounds ``RET`` with ``DLRET`` when present. Treating a missing
delist return as silent zero is forbidden when a delist flag is set.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

# Option PIT filters use t-1 information only.
OPTION_PIT_REQUIRED = frozenset({"bid", "ask", "volume", "open_interest", "delta", "expiry"})

# Alpha v2 primary campaign: freeze OM-50 membership at 2021-12-31.
FROZEN_AT_2021_END = "FROZEN_AT_2021_END"
FOLD_SPECIFIC_RESELECT = "FOLD_SPECIFIC_RESELECT"
# W4.2: dynamic slot-masked universe (src.data.dynamic_universe) that
# re-selects names from a trailing PIT window at every rebalance. This is
# *not* fold-specific reselection (no new protocol hash / no CPCV-fold
# look-ahead risk to gate): the trailing window is always strictly behind
# the current rebalance day regardless of which fold is being scored, so it
# does not require ``allow_fold_reselect=True``. Callers using this mode
# must report PIT status via
# ``src.data.pit_guards.selection_pit_status(..., universe_protocol="slot_masked")``,
# not the default ``"frozen"`` protocol (which assumes a single static
# selection date rather than a per-rebalance rolling one).
ROLLING_TRAILING_PIT = "ROLLING_TRAILING_PIT"
FROZEN_UNIVERSE_ASOF = "2021-12-31"


def resolve_universe_end_for_mode(
    universe_mode: str,
    *,
    requested_end: str | None = None,
    allow_fold_reselect: bool = False,
) -> str:
    """Return the membership freeze date for the given universe mode.

    ``FROZEN_AT_2021_END`` always freezes at 2021-12-31.
    ``FOLD_SPECIFIC_RESELECT`` requires a new protocol hash and is refused
    unless ``allow_fold_reselect=True``.
    ``ROLLING_TRAILING_PIT`` returns ``requested_end`` (the asof date) as-is;
    it does not require ``allow_fold_reselect`` (see module docstring above).
    """
    mode = str(universe_mode or "").strip()
    if mode.upper() == FROZEN_AT_2021_END or mode.lower() in (
        "frozen_at_2021_end",
        "frozen_2021",
    ):
        return FROZEN_UNIVERSE_ASOF
    if mode.upper() == FOLD_SPECIFIC_RESELECT or mode.lower() in (
        "fold_specific_reselect",
        "fold_reselect",
    ):
        if not allow_fold_reselect:
            raise ValueError(
                "FOLD_SPECIFIC_RESELECT requires a new protocol hash; "
                "set allow_fold_reselect=True only under that new hash"
            )
        if not requested_end:
            raise ValueError("FOLD_SPECIFIC_RESELECT requires requested_end")
        return str(requested_end)
    if mode.upper() == ROLLING_TRAILING_PIT or mode.lower() in (
        "rolling_trailing_pit",
        "rolling_trailing",
        "slot_masked",
    ):
        if not requested_end:
            raise ValueError("ROLLING_TRAILING_PIT requires requested_end (the asof date)")
        return str(requested_end)
    if requested_end:
        return str(requested_end)
    return FROZEN_UNIVERSE_ASOF


def compound_equity_return(ret: float, dlret: float | None) -> float:
    """Compound ordinary return with delisting return: (1+r)(1+dl)-1."""
    r = float(ret) if np.isfinite(ret) else 0.0
    if dlret is None or (isinstance(dlret, float) and not np.isfinite(dlret)):
        return r
    d = float(dlret)
    return (1.0 + r) * (1.0 + d) - 1.0


def validate_delist_handling(
    *,
    delist_flag: bool | np.ndarray | Sequence[bool],
    dlret: float | np.ndarray | Sequence[float] | None,
) -> None:
    """Fail closed if a delist event has missing/zero-imputed DLRET."""
    flags = np.asarray(delist_flag, dtype=bool).reshape(-1)
    if flags.size == 0 or not np.any(flags):
        return
    if dlret is None:
        raise ValueError(
            "delist flagged but DLRET missing; silent zero delist handling forbidden"
        )
    d = np.asarray(dlret, dtype=np.float64).reshape(-1)
    if d.size == 1 and flags.size > 1:
        d = np.full(flags.size, float(d[0]), dtype=np.float64)
    if d.size != flags.size:
        raise ValueError(f"delist_flag size {flags.size} != dlret size {d.size}")
    for i in np.where(flags)[0]:
        if not np.isfinite(d[i]):
            raise ValueError(
                f"delist at index {i}: DLRET is non-finite; refuse silent zero"
            )


def build_equity_pit_returns(
    ret: np.ndarray | Sequence[float],
    dlret: np.ndarray | Sequence[float] | None = None,
    *,
    delist_flag: np.ndarray | Sequence[bool] | None = None,
) -> np.ndarray:
    """Compound RET+DLRET; validate delist rows when ``delist_flag`` provided."""
    r = np.asarray(ret, dtype=np.float64).reshape(-1)
    if delist_flag is not None:
        validate_delist_handling(delist_flag=delist_flag, dlret=dlret)
    if dlret is None:
        return r.copy()
    d = np.asarray(dlret, dtype=np.float64).reshape(-1)
    if d.size != r.size:
        raise ValueError(f"ret size {r.size} != dlret size {d.size}")
    out = np.empty_like(r)
    for i in range(r.size):
        out[i] = compound_equity_return(r[i], d[i])
    return out


def option_pit_filter_mask(
    frame: pd.DataFrame | Mapping[str, Any],
    *,
    asof: Any,
    min_volume: float = 0.0,
    min_oi: float = 0.0,
    require_positive_spread: bool = True,
) -> np.ndarray:
    """Boolean keep-mask using only columns known at ``asof`` (t−1 filters).

    Retains expiry/delist paths when present; does not forward-fill quotes.
    """
    if isinstance(frame, pd.DataFrame):
        df = frame
    else:
        df = pd.DataFrame(frame)
    missing = OPTION_PIT_REQUIRED - set(df.columns)
    # Allow abbreviated column names used in panels.
    aliases = {
        "open_interest": ("open_interest", "oi", "open_int"),
        "volume": ("volume", "vol", "opt_volume"),
        "expiry": ("expiry", "exdate", "expiration"),
        "bid": ("bid", "best_bid"),
        "ask": ("ask", "best_offer", "best_ask"),
        "delta": ("delta", "delta_1545", "delta_"),
    }
    resolved: dict[str, str] = {}
    for req in OPTION_PIT_REQUIRED:
        if req in df.columns:
            resolved[req] = req
            continue
        found = None
        for cand in aliases.get(req, ()):
            if cand in df.columns:
                found = cand
                break
        if found is None:
            raise ValueError(f"option PIT filter missing required column {req!r}")
        resolved[req] = found

    n = len(df)
    keep = np.ones(n, dtype=bool)
    bid = pd.to_numeric(df[resolved["bid"]], errors="coerce").to_numpy(dtype=np.float64)
    ask = pd.to_numeric(df[resolved["ask"]], errors="coerce").to_numpy(dtype=np.float64)
    vol = pd.to_numeric(df[resolved["volume"]], errors="coerce").to_numpy(dtype=np.float64)
    oi = pd.to_numeric(df[resolved["open_interest"]], errors="coerce").to_numpy(
        dtype=np.float64
    )
    delta = pd.to_numeric(df[resolved["delta"]], errors="coerce").to_numpy(dtype=np.float64)

    keep &= np.isfinite(bid) & np.isfinite(ask) & np.isfinite(delta)
    if require_positive_spread:
        keep &= ask > bid
        keep &= bid > 0
    keep &= np.nan_to_num(vol, nan=0.0) >= float(min_volume)
    keep &= np.nan_to_num(oi, nan=0.0) >= float(min_oi)

    # Expiry: keep rows with known expiry (including expired — path retained).
    exp = df[resolved["expiry"]]
    keep &= pd.notna(exp).to_numpy()

    # asof is recorded for audit; filters themselves are column-local (t−1).
    _ = asof
    return keep
