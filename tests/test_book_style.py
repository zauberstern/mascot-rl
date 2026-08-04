"""D1: src.reporting.book_style aesthetic-system unit tests."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.reporting.book_style import (
    FAMILY_ORDER,
    FAMILY_PALETTE,
    PdfBook,
    assert_no_default_mpl_colors,
    build_manifest,
    caption,
    family_color,
    require_units_label,
    section_divider,
    stamp_footer,
    stamp_n,
    strategy_family,
    table_figure,
)


def test_strategy_family_classifies_known_names() -> None:
    assert strategy_family("equal_weight") == "naive"
    assert strategy_family("no_trade") == "naive"
    assert strategy_family("cap_weight_bah") == "naive"
    assert strategy_family("policy") == "policy"
    assert strategy_family("happo") == "policy"
    assert strategy_family("olps:pamr") == "olps"
    assert strategy_family("olps:ons") == "olps"
    assert strategy_family("ceiling:kelly_cnn") == "ml_ceiling"
    assert strategy_family("min_variance") == "classical_optimizer"


def test_strategy_family_unknown_falls_back_to_naive() -> None:
    assert strategy_family("some_future_benchmark_xyz") == "naive"


def test_family_color_is_stable_and_in_palette() -> None:
    for name in ("policy", "happo", "equal_weight", "olps:pamr", "ceiling:kelly_cnn"):
        c = family_color(name)
        assert c in FAMILY_PALETTE.values()


def test_family_palette_covers_all_families_and_is_colorblind_distinct() -> None:
    assert set(FAMILY_PALETTE.keys()) == set(FAMILY_ORDER)
    colors = list(FAMILY_PALETTE.values())
    assert len(colors) == len(set(colors)), "family colors must be pairwise distinct"


def test_require_units_label_accepts_parenthetical_units() -> None:
    assert require_units_label("Return (%)") == "Return (%)"
    assert require_units_label("Weight (fraction of NAV)")


def test_require_units_label_rejects_bare_label() -> None:
    with pytest.raises(ValueError):
        require_units_label("Return")
    with pytest.raises(ValueError):
        require_units_label("")


def test_stamp_n_writes_text_onto_axis() -> None:
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    stamp_n(ax, 42)
    texts = [t.get_text() for t in ax.texts]
    assert any("42" in t for t in texts)
    plt.close(fig)


def test_stamp_footer_includes_all_manifest_fields() -> None:
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    manifest = build_manifest(
        cfg={"a": 1},
        estimand_hash="abc123def456",
        scorecard="total_net",
        date_start="2014-01-01",
        date_end="2024-12-31",
    )
    stamp_footer(fig, manifest)
    footer_text = fig.texts[-1].get_text()
    assert "git=" in footer_text
    assert "cfg=" in footer_text
    assert "estimand=" in footer_text
    assert "scorecard=total_net" in footer_text
    assert "2014-01-01" in footer_text and "2024-12-31" in footer_text
    plt.close(fig)


def test_stamp_footer_degrades_gracefully_on_empty_manifest() -> None:
    fig, ax = plt.subplots()
    stamp_footer(fig, {})
    assert fig.texts[-1].get_text() == "provenance unavailable"
    plt.close(fig)


def test_caption_adds_text_to_figure() -> None:
    fig, ax = plt.subplots()
    caption(fig, "This figure shows something important.")
    assert any("important" in t.get_text() for t in fig.texts)
    plt.close(fig)


def test_table_figure_renders_zebra_striped_table() -> None:
    df = pd.DataFrame({"name": ["a", "b", "c"], "sharpe": [1.234, -0.5, 0.0]})
    fig = table_figure(df, "Test table")
    assert fig is not None
    plt.close(fig)


def test_section_divider_renders_title_and_subtitle() -> None:
    fig = section_divider("Section 1: Headline", "Policy vs benchmark panel")
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "Section 1: Headline" in texts
    assert "Policy vs benchmark panel" in texts
    plt.close(fig)


def test_pdfbook_accumulates_pages(tmp_path) -> None:
    out = tmp_path / "book.pdf"
    with PdfBook(out) as book:
        for i in range(3):
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])
            book.add(fig)
            plt.close(fig)
        assert book.n_pages == 3
    assert out.is_file()
    assert out.stat().st_size > 0


def test_assert_no_default_mpl_colors_passes_on_house_palette() -> None:
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], color=FAMILY_PALETTE["policy"])
    ax.plot([3, 2, 1], color=FAMILY_PALETTE["naive"])
    assert_no_default_mpl_colors(fig)
    plt.close(fig)


def test_assert_no_default_mpl_colors_fails_on_tab10_default() -> None:
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], color="#1f77b4")
    with pytest.raises(AssertionError):
        assert_no_default_mpl_colors(fig)
    plt.close(fig)
