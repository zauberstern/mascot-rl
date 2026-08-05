"""Evidence-gated spectrum study package."""
from __future__ import annotations

from src.spectrum.registry import (
    AXES,
    PORTFOLIO_ARM_IDS,
    PORTFOLIO_ARMS,
    Citation,
    SpectrumOption,
    allowed_ids,
    all_options,
    default_id,
    get_option,
    render_markdown_table,
    render_portfolio_arms_table,
    validate_cfg,
    validate_choice,
    validate_portfolio_arm,
)

__all__ = [
    "AXES",
    "PORTFOLIO_ARM_IDS",
    "PORTFOLIO_ARMS",
    "Citation",
    "SpectrumOption",
    "allowed_ids",
    "all_options",
    "default_id",
    "get_option",
    "render_markdown_table",
    "render_portfolio_arms_table",
    "validate_cfg",
    "validate_choice",
    "validate_portfolio_arm",
]
