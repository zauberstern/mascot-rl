"""Unified figure_style across book_style and campaign plots (B4)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mascotrl.reporting.book_style import apply_academic_rc
from mascotrl.reporting.figures.figure_style import apply_figure_rc
from mascotrl.reporting.figures.validate import (
    assert_no_default_mpl_colors,
    run_figure_validators,
)


def test_academic_and_figure_share_prop_cycle() -> None:
    apply_figure_rc()
    figure_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    apply_academic_rc()
    academic_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    assert figure_cycle == academic_cycle


def test_sample_figure_passes_validators_non_strict() -> None:
    from mascotrl.reporting.book_style import FAMILY_PALETTE
    from mascotrl.reporting.figures.figure_style import (
        FIGURE_HEIGHT_DEFAULT_IN,
        FIGURE_WIDTH_FULL_IN,
        apply_figure_rc,
        style_axes,
    )

    apply_figure_rc()
    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTH_FULL_IN, FIGURE_HEIGHT_DEFAULT_IN),
        constrained_layout=True,
    )
    ax.plot([0, 1, 2], [0.0, 0.01, 0.02], color=FAMILY_PALETTE["policy"], label="Policy")
    ax.plot([0, 1, 2], [0.0, 0.005, 0.01], color=FAMILY_PALETTE["naive"], ls="--", label="Equal weight")
    ax.set_xlabel("OOS step (index)")
    ax.set_ylabel("Cumulative net return (fraction)")
    ax.legend(loc="best", frameon=False)
    style_axes(ax, zero_line=True)
    assert_no_default_mpl_colors(fig, stem="sample")
    run_figure_validators(fig, stem="sample", strict=False)
    plt.close(fig)
