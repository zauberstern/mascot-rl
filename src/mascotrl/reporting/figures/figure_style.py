"""Figure figure profile: text-block width, serif, no on-figure captions/footers.

Must not import ``book_style`` caption/footer helpers. Colour palettes for
sleeves/regimes live here so figure renders stay independent of the ops book.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

FIGURE_WIDTH_FULL_IN = 5.906
FIGURE_WIDTH_HALF_IN = 2.86
FIGURE_HEIGHT_DEFAULT_IN = 3.4
FIGURE_HEIGHT_HALF_IN = 2.6
FIGURE_HEIGHT_TALL_IN = 4.6
FIGURE_DPI = 400

MAX_CATEGORY_LABELS = 12  # readable on full-width figure axis

SLEEVE_PALETTE = {
    "trend": "#0072B2",
    "reversal": "#D55E00",
    "carry": "#009E73",
    "defensive": "#56B4E9",
    "lottery": "#CC79A7",
    "illiquid": "#E69F00",
    "core": "#999999",
}

# Okabe-Ito 8-color categorical palette (colorblind-safe).
OKABE_ITO: tuple[str, ...] = (
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
)

# Frozen archetype colour + marker grammar (must match across all B/T figures).
ARCHETYPE_COLOR_MAP: dict[str, str] = {
    "Cheetah": "#E69F00",
    "Fox": "#56B4E9",
    "Tortoise": "#009E73",
    "Magpie": "#D55E00",
    "Hummingbird": "#CC79A7",
    "Owl": "#000000",
    "trend_follower": "#E69F00",
    "contrarian": "#56B4E9",
    "risk_manager": "#009E73",
    "speculator": "#D55E00",
    "tactical_rotator": "#CC79A7",
    "mixed": "#000000",
}

ARCHETYPE_MARKER_MAP: dict[str, str] = {
    "Cheetah": "o",
    "Fox": "s",
    "Tortoise": "^",
    "Magpie": "D",
    "Hummingbird": "v",
    "Owl": "X",
    "trend_follower": "o",
    "contrarian": "s",
    "risk_manager": "^",
    "speculator": "D",
    "tactical_rotator": "v",
    "mixed": "X",
}

REGIME_PALETTE = {
    "calm": "#F2F2F2",
    "inflationary": "#FDE7C8",
    "crisis": "#F6D6D6",
}

_LINE_STYLES = ("-", "--", "-.", ":")


def apply_figure_rc() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.2,
            "patch.linewidth": 0.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.constrained_layout.use": True,
            "axes.prop_cycle": mpl.cycler(
                color=[
                    "#D55E00",
                    "#0072B2",
                    "#009E73",
                    "#CC79A7",
                    "#E69F00",
                    "#56B4E9",
                    "#000000",
                ]
            ),
        }
    )


def greyscale_safe_styles(n: int) -> list[str]:
    """Cycle line styles so multi-series figures stay separable in greyscale."""
    n = max(int(n), 0)
    if n == 0:
        return []
    return [_LINE_STYLES[i % len(_LINE_STYLES)] for i in range(n)]


def empty_axes_note(ax: Any, message: str) -> None:
    """Centred explanation when data is missing or degenerate (never a blank chart)."""
    ax.cla()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.text(
        0.5,
        0.5,
        str(message),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        wrap=True,
    )


def shade_regimes(ax: Any, dates: Any, regime_series: Any) -> None:
    """Pale background bands from a regime label series aligned to ``dates``."""
    import numpy as np

    if dates is None or regime_series is None:
        return
    dates = list(dates)
    regimes = list(regime_series)
    if not dates or len(dates) != len(regimes):
        return
    i = 0
    n = len(dates)
    while i < n:
        lab = str(regimes[i] or "calm").lower()
        j = i + 1
        while j < n and str(regimes[j] or "calm").lower() == lab:
            j += 1
        color = REGIME_PALETTE.get(lab, REGIME_PALETTE["calm"])
        x0 = dates[i]
        x1 = dates[min(j, n - 1)]
        try:
            ax.axvspan(x0, x1, color=color, alpha=0.9, zorder=0, lw=0)
        except (TypeError, ValueError):
            pass
        i = j


def _resolve_width(width: str | float) -> float:
    if isinstance(width, (int, float)):
        return float(width)
    key = str(width).lower()
    if key in ("full", "textwidth"):
        return FIGURE_WIDTH_FULL_IN
    if key in ("half", "halfwidth"):
        return FIGURE_WIDTH_HALF_IN
    return FIGURE_WIDTH_FULL_IN


@contextmanager
def figure_block(
    width: str | float = "full",
    height_in: float | None = None,
    nrows: int = 1,
    ncols: int = 1,
    **subplots_kw: Any,
) -> Iterator[tuple[Any, Any]]:
    """Yield ``(fig, ax_or_axes)`` with figure rc and constrained_layout."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_figure_rc()
    w = _resolve_width(width)
    if height_in is None:
        height_in = (
            FIGURE_HEIGHT_HALF_IN if w <= FIGURE_WIDTH_HALF_IN + 0.01 else FIGURE_HEIGHT_DEFAULT_IN
        )
    h = float(min(max(float(height_in), 2.0), 8.0))
    subplots_kw = dict(subplots_kw)
    subplots_kw.setdefault("constrained_layout", True)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(w, h), **subplots_kw)
    try:
        yield fig, ax
    finally:
        plt.close(fig)


def save_figure(
    fig: Any,
    stem: str | Path,
    *,
    pdf: bool = True,
    strict: bool = True,
    skip_validators: bool = False,
) -> list[str]:
    """Validate (optional) then write PNG at 400 dpi and optional PDF twin."""
    from src.reporting.figures.validate import run_figure_validators

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    label = stem.name
    if not skip_validators:
        run_figure_validators(fig, stem=label, strict=strict)

    written: list[str] = []
    png = stem if stem.suffix.lower() == ".png" else stem.with_suffix(".png")
    fig.savefig(
        png,
        dpi=FIGURE_DPI,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    written.append(str(png))
    if pdf:
        pdf_path = png.with_suffix(".pdf")
        fig.savefig(
            pdf_path,
            dpi=FIGURE_DPI,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.05,
        )
        written.append(str(pdf_path))
    return written


def thin_date_axis(ax: Any, *, maj: str = "year") -> None:
    """AutoDateLocator + ConciseDateFormatter (or yearly). Prevent black-line date axes."""
    import matplotlib.dates as mdates

    if maj == "year":
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def thin_category_axis(
    ax: Any, *, axis: str = "y", max_labels: int = MAX_CATEGORY_LABELS
) -> None:
    """If too many category ticks, keep evenly spaced subset; rotate if x."""
    import numpy as np

    if axis == "x":
        ticks = list(ax.get_xticks())
        labels = [t.get_text() for t in ax.get_xticklabels()]
    else:
        ticks = list(ax.get_yticks())
        labels = [t.get_text() for t in ax.get_yticklabels()]
    n = len(ticks)
    if n <= max_labels:
        if axis == "x" and n > 6:
            for lab in ax.get_xticklabels():
                lab.set_rotation(30)
                lab.set_ha("right")
        return
    idx = np.linspace(0, n - 1, max_labels, dtype=int)
    new_ticks = [ticks[i] for i in idx]
    new_labels = [labels[i] for i in idx]
    if axis == "x":
        ax.set_xticks(new_ticks)
        ax.set_xticklabels(new_labels, rotation=30, ha="right")
    else:
        ax.set_yticks(new_ticks)
        ax.set_yticklabels(new_labels)


def rank_truncate(
    items: list,
    *,
    key: Any,
    n: int = MAX_CATEGORY_LABELS,
    reverse: bool = True,
) -> tuple[list, int]:
    """Return top-n items by key and count of omitted. Caller annotates '+N more'."""
    if len(items) <= n:
        return list(items), 0
    ranked = sorted(items, key=key, reverse=reverse)
    return ranked[:n], len(items) - n


def subsample_heatmap_axes(
    matrix: Any,
    row_labels: list,
    col_labels: list,
    *,
    max_labels: int = MAX_CATEGORY_LABELS,
) -> tuple[Any, list, list, int]:
    """Truncate rows/cols by mean |value| when either axis exceeds max_labels."""
    import numpy as np

    m = np.asarray(matrix, dtype=float)
    omitted = 0
    if len(row_labels) > max_labels:
        row_imp = np.nanmean(np.abs(m), axis=1)
        top = sorted(np.argsort(-row_imp)[:max_labels])
        m = m[top]
        row_labels = [row_labels[i] for i in top]
        omitted += len(row_imp) - max_labels
    if len(col_labels) > max_labels:
        col_imp = np.nanmean(np.abs(m), axis=0)
        top = sorted(np.argsort(-col_imp)[:max_labels])
        m = m[:, top]
        col_labels = [col_labels[i] for i in top]
        omitted += len(col_imp) - max_labels
    return m, row_labels, col_labels, omitted


def annotate_truncation(
    ax: Any,
    omitted: int,
    *,
    corner: str = "bottom_right",
) -> None:
    """Corner note when rank_truncate dropped items."""
    if omitted <= 0:
        return
    corners = {
        "bottom_right": ((0.98, 0.02), "right", "bottom"),
        "top_left": ((0.02, 0.98), "left", "top"),
        "top_right": ((0.98, 0.98), "right", "top"),
    }
    xy, ha, va = corners.get(corner, corners["bottom_right"])
    ax.annotate(
        f"+{omitted} more",
        xy=xy,
        xycoords="axes fraction",
        ha=ha,
        va=va,
        fontsize=7,
    )


def style_axes(ax: Any, *, zero_line: bool = False, date_axis: bool = False) -> None:
    """House grid / spine / optional zero line for figure axes."""
    ax.grid(True, axis="y", color="#E6E6E6", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    if zero_line:
        ax.axhline(0.0, color="#222222", lw=0.8, zorder=2)
    if date_axis:
        thin_date_axis(ax)
