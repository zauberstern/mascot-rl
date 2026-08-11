"""PIT guards for universe selection vs evaluation windows.

Frozen universe: eval must start after selection ends. Slot-masked fixed-K:
PIT-eligible names fill slots each rebalance; inactive slots zeroed.
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
    """PIT-clean when eval starts after universe selection (or slot-masked)."""
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
    """Filter rows to index members; empty ``members`` passes through with disclosure."""
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
