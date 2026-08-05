"""Part F: figure validators."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from mascotrl.reporting.book_style import FAMILY_PALETTE
from mascotrl.reporting.figures.validate import (
    assert_axis_labels_human,
    assert_greyscale_separable,
    assert_legend_present,
    assert_no_default_mpl_colors,
    assert_no_overlap,
    assert_no_raw_identifiers,
    assert_within_canvas,
    run_figure_validators,
)


def _house_line(ax, ys, *, color, ls="-", label="A") -> None:
    ax.plot([0, 1, 2], ys, color=color, ls=ls, label=label)


def test_overlap_title_and_caption_fails() -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1], color=FAMILY_PALETTE["policy"])
    ax.set_title("Title that will collide", pad=0)
    fig.text(0.5, 0.92, "Caption that overlaps the title", ha="center", fontsize=14)
    fig.canvas.draw()
    with pytest.raises(AssertionError, match="overlap"):
        assert_no_overlap(fig, stem="overlap_demo", tol_px=0.0)
    plt.close(fig)


def test_clipped_rotated_tick_fails() -> None:
    # Force clipping: tiny canvas, huge rotated labels, no layout engine.
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig = plt.figure(figsize=(2.2, 1.6), dpi=100)
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])  # leaves little bottom margin
    ax.bar([0, 1, 2], [1, 2, 3], color=FAMILY_PALETTE["policy"])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(
        [
            "ExtremelyLongTickLabelAlphaXXXX",
            "ExtremelyLongTickLabelBetaXXXX",
            "ExtremelyLongTickLabelGammaXXXX",
        ],
        rotation=80,
        ha="right",
        fontsize=18,
    )
    fig.canvas.draw()
    with pytest.raises(AssertionError, match="clipped|outside"):
        assert_within_canvas(fig, stem="clip_demo", pad_px=0.0)
    plt.close(fig)


def test_snake_case_tick_fails() -> None:
    fig, ax = plt.subplots()
    ax.bar([0, 1], [1, 2], color=FAMILY_PALETTE["policy"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["equal_weight", "no_trade"])
    with pytest.raises(AssertionError, match="snake_case|raw"):
        assert_no_raw_identifiers(fig, stem="snake_demo")
    plt.close(fig)


def test_default_mpl_color_fails() -> None:
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], color="#1f77b4")  # classic tab10 blue
    with pytest.raises(AssertionError, match="default matplotlib"):
        assert_no_default_mpl_colors(fig, stem="color_demo")
    plt.close(fig)


def test_greyscale_four_lines_one_style_fails() -> None:
    fig, ax = plt.subplots()
    for i, c in enumerate(
        [
            FAMILY_PALETTE["policy"],
            FAMILY_PALETTE["naive"],
            FAMILY_PALETTE["classical_optimizer"],
            FAMILY_PALETTE["olps"],
        ]
    ):
        ax.plot([0, 1], [i, i + 1], color=c, ls="-", label=f"Series {i}")
    with pytest.raises(AssertionError, match="greyscale|styles"):
        assert_greyscale_separable(fig, stem="grey_demo")
    plt.close(fig)


def test_compliant_figure_passes_all_seven() -> None:
    fig, ax = plt.subplots(figsize=(5.9, 3.4), constrained_layout=True)
    styles = ["-", "--", "-.", ":"]
    colors = [
        FAMILY_PALETTE["policy"],
        FAMILY_PALETTE["naive"],
        FAMILY_PALETTE["classical_optimizer"],
        FAMILY_PALETTE["olps"],
    ]
    for i, (c, ls) in enumerate(zip(colors, styles)):
        ax.plot([0, 1, 2], [i, i + 0.5, i + 1], color=c, ls=ls, label=f"Series {i}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative net return (fraction of initial NAV)")
    ax.legend(frameon=False, loc="best")
    run_figure_validators(fig, stem="compliant", strict=True)
    plt.close(fig)


def test_axis_labels_human_rejects_bare() -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color=FAMILY_PALETTE["policy"])
    ax.set_ylabel("Return")
    with pytest.raises(AssertionError):
        assert_axis_labels_human(fig)
    plt.close(fig)


def test_legend_required_for_multiple_labelled() -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color=FAMILY_PALETTE["policy"], label="A")
    ax.plot([0, 1], [1, 0], color=FAMILY_PALETTE["naive"], label="B")
    with pytest.raises(AssertionError, match="legend"):
        assert_legend_present(fig)
    plt.close(fig)
