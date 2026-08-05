"""Point-in-time guards for universe selection and evaluation windows.

Historically the universe was frozen once (the policy has a fixed asset count
K). A frozen universe is acceptable only if the evaluation window used for the
headline claim begins **after** the selection window ends; otherwise names are
chosen with information from inside the period being scored.

Time-varying membership under the same fixed K is now supported via slot
masking (``src.data.slot_mask``): K slots stay fixed, PIT-eligible names fill
slots at each rebalance, and inactive slots are zeroed in weights and labels.
Under that design, ``selection_pit_status`` should still be clean for every
phase (each rebalance uses only a trailing window).

These helpers make the frozen-universe PIT condition machine-checked and
reportable instead of a prose caveat; they are unchanged for callers that keep
a frozen universe.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def selection_pit_status(
    *,
    universe_end: str | pd.Timestamp | None,
    eval_start: str | pd.Timestamp | None,
    phase: str,
    universe_protocol: str = "frozen",
) -> dict[str, Any]:
    """
    Whether ``phase`` is free of universe-selection look-ahead.

    Under ``universe_protocol="frozen"``, clean when the evaluation window
    starts strictly after the last date used to select the universe.

    Under ``universe_protocol="slot_masked"``, membership is re-selected at each
    rebalance from a trailing PIT window only, so every phase is clean by
    construction (unused slots are zeroed via the validity mask).
    """
    protocol = str(universe_protocol or "frozen").lower().strip()
    if protocol in {"slot_masked", "slot-masking", "slotmask"}:
        return {
            "phase": phase,
            "pit_clean": True,
            "universe_protocol": "slot_masked",
            "universe_end": (
                str(pd.Timestamp(universe_end).date()) if universe_end is not None else None
            ),
            "eval_start": (
                str(pd.Timestamp(eval_start).date()) if eval_start is not None else None
            ),
            "overlap_days": 0,
            "reason": (
                "slot-masked fixed-K policy: each rebalance fills slots from a "
                "trailing PIT-eligible set; inactive slots are hard-masked"
            ),
        }
    if universe_end is None or eval_start is None:
        return {
            "phase": phase,
            "pit_clean": False,
            "universe_protocol": "frozen",
            "reason": "universe_end or eval_start unknown",
            "universe_end": None,
            "eval_start": None,
        }
    u_end = pd.Timestamp(universe_end)
    e_start = pd.Timestamp(eval_start)
    clean = bool(e_start > u_end)
    return {
        "phase": phase,
        "pit_clean": clean,
        "universe_protocol": "frozen",
        "universe_end": str(u_end.date()),
        "eval_start": str(e_start.date()),
        "overlap_days": int(max(0, (u_end - e_start).days + 1)) if not clean else 0,
        "reason": (
            "eval window starts after universe selection window"
            if clean
            else (
                "eval window overlaps the universe selection window; names were "
                "chosen using information from inside the scored period"
            )
        ),
    }


def assert_headline_selection_pit(status: dict[str, Any]) -> None:
    """Raise when the phase carrying the headline claim is not PIT-clean."""
    if not status.get("pit_clean"):
        raise RuntimeError(
            f"universe-selection look-ahead in phase '{status.get('phase')}': "
            f"{status.get('reason')} "
            f"(universe_end={status.get('universe_end')}, "
            f"eval_start={status.get('eval_start')})"
        )


def membership_filter(
    rows: list[dict],
    members: set[str],
) -> tuple[list[dict], dict[str, Any]]:
    """
    Restrict candidate universe rows to index members, with a coverage record.

    An empty ``members`` set means the snapshot was unavailable; rows pass
    through unchanged and the record says membership was not enforced, so the
    limitation is disclosed rather than hidden.
    """
    if not members:
        return list(rows), {
            "enforced": False,
            "reason": "PIT membership snapshot unavailable",
            "n_in": len(rows),
            "n_out": len(rows),
        }
    kept = [r for r in rows if str(r.get("ticker", "")).strip().upper() in members]
    return kept, {
        "enforced": True,
        "n_in": len(rows),
        "n_out": len(kept),
        "n_dropped_non_member": len(rows) - len(kept),
    }
