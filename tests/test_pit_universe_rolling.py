"""W4.2: ROLLING_TRAILING_PIT universe mode (slot-masked, per-rebalance)."""
from __future__ import annotations

import pytest

from src.features.pit_universe import (
    FOLD_SPECIFIC_RESELECT,
    FROZEN_AT_2021_END,
    ROLLING_TRAILING_PIT,
    resolve_universe_end_for_mode,
)


def test_rolling_trailing_pit_returns_requested_end_without_fold_reselect_flag():
    end = resolve_universe_end_for_mode(
        ROLLING_TRAILING_PIT,
        requested_end="2019-06-30",
        allow_fold_reselect=False,
    )
    assert end == "2019-06-30"


def test_rolling_trailing_pit_requires_requested_end():
    with pytest.raises(ValueError):
        resolve_universe_end_for_mode(ROLLING_TRAILING_PIT, requested_end=None)


def test_rolling_trailing_pit_is_not_the_fold_reselect_gate():
    """Unlike FOLD_SPECIFIC_RESELECT, rolling-trailing never needs the flag."""
    with pytest.raises(ValueError):
        resolve_universe_end_for_mode(
            FOLD_SPECIFIC_RESELECT, requested_end="2019-06-30", allow_fold_reselect=False
        )
    # Same requested_end succeeds under ROLLING_TRAILING_PIT with the flag
    # left at its default False.
    end = resolve_universe_end_for_mode(ROLLING_TRAILING_PIT, requested_end="2019-06-30")
    assert end == "2019-06-30"


def test_frozen_mode_unaffected_by_new_mode():
    assert resolve_universe_end_for_mode(FROZEN_AT_2021_END) == "2021-12-31"


def test_rolling_trailing_pit_selection_pit_status_uses_slot_masked_protocol():
    from src.data.pit_guards import selection_pit_status

    status = selection_pit_status(
        universe_end="2019-06-30",
        eval_start="2019-01-01",
        phase="oos",
        universe_protocol="slot_masked",
    )
    assert status["pit_clean"] is True
    assert status["universe_protocol"] == "slot_masked"
