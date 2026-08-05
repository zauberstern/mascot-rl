"""Equity-core feature blocks (returns / vol / liquidity / normalize / assemble)."""
from __future__ import annotations

from mascotrl.features.blocks.assemble import assemble_equity_feature_cube
from mascotrl.features.blocks.normalize import (
    cross_sectional_zscore,
    expanding_causal_zscore,
    normalize_cross_section_panel,
    winsorize_cross_section,
    winsorize_panel,
)
from mascotrl.features.blocks.obs_builder import PanelObservationBuilder
from mascotrl.features.blocks.portfolio_state import build_portfolio_state_features

__all__ = [
    "PanelObservationBuilder",
    "assemble_equity_feature_cube",
    "build_portfolio_state_features",
    "cross_sectional_zscore",
    "expanding_causal_zscore",
    "normalize_cross_section_panel",
    "winsorize_cross_section",
    "winsorize_panel",
]
