"""Doctrine locks: OM IV / vsurfd law and surface lane names."""
from __future__ import annotations

from pathlib import Path

from mascotrl.data.duckdb_engine import OptionFilterConfig
from mascotrl.data.surface_signals import (
    GRID_POINTS_PER_DAY,
    KELLY_DELTAS_CALL,
    KELLY_DELTAS_PUT,
    KELLY_TENORS,
)
from mascotrl.spectrum.cell_schema import SCHEMA


def test_use_surface_signals_in_schema():
    assert "use_surface_signals" in SCHEMA


def test_kelly_grid_matches_om_delta_tenor_counts():
    assert len(KELLY_TENORS) == 11
    assert len(KELLY_DELTAS_PUT) == 17
    assert len(KELLY_DELTAS_CALL) == 17
    assert GRID_POINTS_PER_DAY == len(KELLY_TENORS) * (
        len(KELLY_DELTAS_PUT) + len(KELLY_DELTAS_CALL)
    )


def test_option_filter_defaults_require_iv_and_attrition_arb():
    cfg = OptionFilterConfig()
    assert cfg.require_iv is True
    assert cfg.drop_surface_arb_days is False


