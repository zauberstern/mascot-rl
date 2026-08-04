"""Book-grade aesthetic system: unified matplotlib profile for ops book, core suite, tearsheets.

Single visual identity: ``apply_academic_rc`` delegates to ``figure_style.apply_figure_rc``.
Includes Okabe-Ito color constants, figure helpers, family palette, validators, and
``save_pdf_png`` / ``use_agg`` for spectrum figure suites.
"""
from __future__ import annotations

import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

# Comfortable screen canvases (inches) — not journal mm stubs.
WIDTH_SINGLE_IN = 7.5
WIDTH_DOUBLE_IN = 11.0
HEIGHT_DEFAULT = 4.2
HEIGHT_SINGLE = 4.0

# Okabe-Ito aligned aliases (match FAMILY_PALETTE without import cycle).
C_NAVY = "#0072B2"
C_BLUE = "#D55E00"
C_STEEL = "#6B6B6B"
C_GRAY = "#6B6B6B"
C_LIGHT = "#D9D9D9"
C_IS_SHADE = "#E6E6E6"
C_POS = "#009E73"
C_NEG = "#CC79A7"
C_ACCENT = "#D55E00"
C_ZERO = "#222222"

CMAP_DIVERGING = "RdBu_r"
CMAP_SEQUENTIAL = "Greys"


def apply_academic_rc() -> None:
    """Delegate to figure Okabe-Ito serif profile (single visual identity)."""
    from src.reporting.figures.figure_style import apply_figure_rc

    apply_figure_rc()


@contextmanager
def academic_figure(
    width: str = "double",
    height_in: float | None = None,
    nrows: int = 1,
    ncols: int = 1,
    *,
    legend_space: bool = False,
    **subplots_kw,
) -> Iterator[tuple]:
    """Yield (fig, ax_or_axes). ``legend_space`` reserves a right margin for legends."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_academic_rc()
    w = WIDTH_DOUBLE_IN if width == "double" else WIDTH_SINGLE_IN
    if legend_space:
        w += 1.6
    h = height_in if height_in is not None else (
        HEIGHT_SINGLE if width == "single" else HEIGHT_DEFAULT
    )
    h = float(min(max(h, 2.5), 8.0))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(w, h), **subplots_kw)
    try:
        yield fig, ax
    finally:
        plt.close(fig)


def shade_insample(ax, x0: float, x1: float, label: str | None = "In-sample") -> None:
    """Gray band for training / in-sample regime."""
    if x1 <= x0:
        return
    ax.axvspan(x0, x1, color=C_IS_SHADE, alpha=0.85, zorder=0, lw=0)
    if label:
        ax.text(
            0.02,
            0.96,
            label,
            transform=ax.transAxes,
            fontsize=8,
            color=C_GRAY,
            va="top",
            ha="left",
        )


def mark_oos_boundary(ax, x: float, label: str = "OOS") -> None:
    ax.axvline(x, color=C_ZERO, ls="--", lw=0.9, zorder=3)
    try:
        x_frac = ax.transAxes.inverted().transform(ax.transData.transform((x, 0.0)))[0]
    except Exception:
        x_frac = 0.5
    x_frac = float(min(max(x_frac, 0.02), 0.92))
    ax.text(
        x_frac,
        1.01,
        label,
        transform=ax.transAxes,
        fontsize=8,
        color=C_ZERO,
        va="bottom",
        ha="left",
    )


def place_legend(ax, fig=None, *, loc: str = "outside right", ncol: int = 1, **kwargs):
    """Place legend outside the data area so it cannot cover series."""
    kw = dict(frameon=False, fancybox=False, framealpha=0.0, **kwargs)
    if loc == "outside right":
        return ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            ncol=ncol,
            **kw,
        )
    if loc == "outside top":
        return ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.02),
            borderaxespad=0.0,
            ncol=ncol,
            **kw,
        )
    return ax.legend(loc=loc, **kw)


def finalize_figure(fig, *, legend_space: bool = False) -> None:
    """Safe layout pass — never let tight_layout blow up the canvas."""
    try:
        if legend_space:
            fig.subplots_adjust(right=0.78, left=0.08, top=0.90, bottom=0.12)
        else:
            fig.tight_layout(pad=0.6)
    except Exception:
        fig.subplots_adjust(left=0.10, right=0.96, top=0.90, bottom=0.12)


def save_figure(fig, path, *, dpi: int = 160, pdf: bool = False, strict: bool = False) -> list[str]:
    """Write PNG only by default. Returns written paths."""
    from src.reporting.figures.validate import run_figure_validators

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.stem if path.suffix else str(path)
    try:
        run_figure_validators(fig, stem=stem, strict=strict)
    except AssertionError:
        if strict:
            raise
    written: list[str] = []
    png = path if path.suffix.lower() == ".png" else path.with_suffix(".png")
    fig.savefig(png, dpi=dpi, facecolor="white", bbox_inches=None, pad_inches=0.15)
    written.append(str(png))
    if pdf:
        pdf_path = png.with_suffix(".pdf")
        fig.savefig(pdf_path, dpi=dpi, facecolor="white", bbox_inches=None)
        written.append(str(pdf_path))
    stale = png.with_suffix(".pdf")
    if not pdf and stale.is_file():
        try:
            stale.unlink()
        except OSError:
            pass
    return written


# Colorblind-safe (Okabe-Ito derived) family palette. Every strategy in the
# book is classified into exactly one family so the reader learns one color
# per class instead of re-decoding ~40 distinct hues across the book.
FAMILY_PALETTE: dict[str, str] = {
    "policy": "#D55E00",  # vermillion — the thing being evaluated
    "naive": "#6B6B6B",  # gray — equal_weight / no_trade / buy_and_hold
    "classical_optimizer": "#0072B2",  # blue — min_variance / risk_parity / mean_variance
    "olps": "#009E73",  # green — pamr / ons / eg / ... online portfolio selection
    "ml_ceiling": "#CC79A7",  # pink — kelly_cnn / ridge_composite / ...
}
FAMILY_ORDER = ("policy", "naive", "classical_optimizer", "olps", "ml_ceiling")

ARM_COLORS = {
    "opt": FAMILY_PALETTE["policy"],
    "eq": FAMILY_PALETTE["olps"],
    "mix": FAMILY_PALETTE["ml_ceiling"],
}

ARM_ORDER = ("opt", "eq", "mix")


def use_agg() -> None:
    import matplotlib

    matplotlib.use("Agg")
    apply_academic_rc()


def save_pdf_png(
    fig: Any, stem: Path, *, dpi: int = 160, pdf: bool = False
) -> dict[str, str]:
    """Write PNG next to ``stem``; optionally also PDF (default off)."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = save_figure(fig, stem.with_suffix(".png"), dpi=dpi, pdf=bool(pdf))
    out: dict[str, str] = {}
    for p in written:
        pp = Path(p)
        if pp.suffix.lower() == ".pdf":
            out["pdf"] = str(pp)
        else:
            out["png"] = str(pp)
    return out


def mirror_files(paths: dict[str, str], dest_dir: Path) -> dict[str, str]:
    """Copy written figure files into ``dest_dir``; return dest paths."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    mirrored: dict[str, str] = {}
    for kind, src in paths.items():
        sp = Path(src)
        if not sp.is_file():
            continue
        dp = dest_dir / sp.name
        shutil.copy2(sp, dp)
        mirrored[kind] = str(dp)
    return mirrored


_FAMILY_NAME_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("policy", re.compile(r"^(policy|happo|ppo|research_policy)\b", re.I)),
    (
        "naive",
        re.compile(r"^(equal_weight|no_trade|buy_and_hold|cap_weight_bah)$", re.I),
    ),
    (
        "classical_optimizer",
        re.compile(
            r"^(min_variance|mean_variance|risk_parity|max_diversification|hrp)\b",
            re.I,
        ),
    ),
    (
        "olps",
        re.compile(r"^(olps[:_]|pamr|ons|eg|anticor|cornk?|rmr|wmamr|up\b)", re.I),
    ),
    (
        "ml_ceiling",
        re.compile(r"^(ceiling[:_]|kelly|ridge_composite|ridge_signal)", re.I),
    ),
)


def strategy_family(name: str) -> str:
    """Classify a strategy id into one of ``FAMILY_ORDER``; defaults to naive.

    Matching is prefix/pattern based against the naming conventions already
    used across ``src/eval/benchmark_panel.py`` and ``src/eval/ceiling_arms.py``
    (``olps:<algo>``, ``ceiling:<algo>``, bare benchmark ids). Unrecognized
    names fall back to ``naive`` rather than raising, since the book must
    keep rendering when a new benchmark is added upstream before this map is.
    """
    n = str(name or "").strip()
    for family, pattern in _FAMILY_NAME_PATTERNS:
        if pattern.search(n):
            return family
    return "naive"


def family_color(name: str) -> str:
    return FAMILY_PALETTE[strategy_family(name)]


def family_legend_handles():
    """Proxy Line2D handles for a family-level legend (one entry per class)."""
    from matplotlib.lines import Line2D

    return [
        Line2D([0], [0], color=FAMILY_PALETTE[f], lw=2.5, label=f.replace("_", " "))
        for f in FAMILY_ORDER
    ]


def require_units_label(label: str) -> str:
    """Enforce D1's "every axis labelled with units" rule.

    A unit-bearing label must carry a parenthetical, e.g. ``"Return (%)"`` or
    ``"Weight (fraction of NAV)"``; a bare ``"Return"`` is rejected so a
    figure cannot silently ship without stating what its axis measures.
    """
    lab = str(label or "").strip()
    if not lab or "(" not in lab or ")" not in lab:
        raise ValueError(
            f"axis label {label!r} must state units in parentheses, e.g. 'Return (%)'"
        )
    return lab


def stamp_n(ax, n: int, *, loc: str = "lower right", fmt: str = "n={n}") -> None:
    """D1 hard rule: every figure states its sample size."""
    text = fmt.format(n=int(n))
    xy = {
        "lower right": (0.98, 0.02, "right", "bottom"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "upper right": (0.98, 0.98, "right", "top"),
    }.get(loc, (0.98, 0.02, "right", "bottom"))
    x, y, ha, va = xy
    ax.text(
        x, y, text, transform=ax.transAxes, fontsize=7, color=C_GRAY, ha=ha, va=va,
    )


def stamp_footer(fig, manifest: Mapping[str, Any]) -> None:
    """Bind a figure to its exact evidence: git sha, config sha, estimand
    hash, scorecard column, and date range, in 6pt across the bottom margin.

    ``manifest`` keys read (all optional, rendered as ``field=value``):
    ``git_sha``, ``config_sha``, ``estimand_hash``, ``scorecard``,
    ``date_start``, ``date_end``.
    """
    parts = []
    git_sha = manifest.get("git_sha")
    if git_sha:
        parts.append(f"git={str(git_sha)[:10]}")
    cfg_sha = manifest.get("config_sha")
    if cfg_sha:
        parts.append(f"cfg={str(cfg_sha)[:10]}")
    est_hash = manifest.get("estimand_hash")
    if est_hash:
        parts.append(f"estimand={str(est_hash)[:10]}")
    scorecard = manifest.get("scorecard")
    if scorecard:
        parts.append(f"scorecard={scorecard}")
    d0, d1 = manifest.get("date_start"), manifest.get("date_end")
    if d0 or d1:
        parts.append(f"dates={d0 or '?'}..{d1 or '?'}")
    text = " | ".join(parts) if parts else "provenance unavailable"
    fig.text(
        0.5, 0.008, text, ha="center", va="bottom", fontsize=6, color=C_GRAY,
    )


def caption(fig, text: str, *, y: float = 0.965) -> None:
    """Narrative caption above the figure body (below any suptitle)."""
    fig.text(
        0.5, y, text, ha="center", va="top", fontsize=8, color=C_GRAY, wrap=True,
    )


def build_manifest(
    *,
    cfg: Mapping[str, Any] | None = None,
    estimand_hash: str | None = None,
    scorecard: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
) -> dict[str, Any]:
    """Assemble the ``stamp_footer`` manifest once per report, so every page
    in the book carries an identical footer without re-deriving git/config
    hashes per figure."""
    from src.reporting.provenance import _git, config_hash

    return {
        "git_sha": _git("rev-parse", "--short", "HEAD"),
        "config_sha": config_hash(dict(cfg)) if cfg else None,
        "estimand_hash": estimand_hash,
        "scorecard": scorecard,
        "date_start": date_start,
        "date_end": date_end,
    }


C_IS_SHADE = "#F2F2F2"


def table_figure(
    df,
    title: str,
    *,
    width: str = "double",
    max_rows: int = 30,
    float_fmt: str = "{:.4f}",
):
    """Render a DataFrame as a zebra-striped, right-aligned-numerics table figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    apply_academic_rc()
    d = df.head(max_rows).copy()
    n_rows, n_cols = d.shape
    w = WIDTH_DOUBLE_IN if width == "double" else WIDTH_SINGLE_IN
    h = float(min(max(0.35 * (n_rows + 2), 2.0), 10.0))
    fig, ax = plt.subplots(figsize=(w, h))
    ax.axis("off")
    ax.set_title(title, fontsize=11, loc="left", color=C_ZERO)

    if n_rows == 0:
        # matplotlib's ax.table() cannot render zero-row cellText; an empty
        # result (e.g. a genuinely empty signal allowlist) is still real
        # evidence and must render, not crash the whole book.
        ax.text(
            0.5, 0.5, "(no rows)", ha="center", va="center", fontsize=10, color=C_GRAY,
        )
        fig.tight_layout(pad=0.6)
        return fig

    def _fmt(v: Any) -> str:
        if isinstance(v, (int, np.integer)):
            return str(v)
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return "NaN"
            return float_fmt.format(v)
        return str(v)

    cell_text = [[_fmt(v) for v in row] for row in d.itertuples(index=False)]
    col_labels = [str(c) for c in d.columns]
    tbl = ax.table(
        cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="right",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.3)
    numeric_cols = {
        i for i, c in enumerate(d.columns) if pd.api.types.is_numeric_dtype(d[c])
    }
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(C_LIGHT)
        if row == 0:
            cell.set_facecolor(C_NAVY)
            cell.set_text_props(color="white", weight="bold")
            cell.set_text_props(ha="center")
        else:
            cell.set_facecolor(C_IS_SHADE if row % 2 == 0 else "white")
            if col in numeric_cols:
                cell.set_text_props(ha="right")
            else:
                cell.set_text_props(ha="left")
    fig.tight_layout(pad=0.6)
    return fig


def section_divider(title: str, subtitle: str = ""):
    """Full-bleed section title page for the multi-page book."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_academic_rc()
    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE_IN, HEIGHT_DEFAULT + 2.0))
    ax.axis("off")
    ax.text(
        0.5, 0.58, title, ha="center", va="center", fontsize=22, color=C_NAVY,
        weight="bold",
    )
    if subtitle:
        ax.text(
            0.5, 0.44, subtitle, ha="center", va="center", fontsize=11, color=C_GRAY,
        )
    ax.axhline(0.5, xmin=0.35, xmax=0.65, color=C_ACCENT, lw=1.5)
    return fig


@contextmanager
def PdfBook(path: str | Path) -> Iterator[Any]:
    """Context manager accumulating every rendered figure into one book.pdf.

    Usage::

        with PdfBook(out_dir / "book.pdf") as book:
            book.add(fig1)
            book.add(fig2)
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = PdfPages(str(path))

    class _Book:
        def __init__(self, pages: "PdfPages") -> None:
            self._pages = pages
            self.n_pages = 0

        def add(self, fig) -> None:
            self._pages.savefig(fig, facecolor="white")
            self.n_pages += 1

    book = _Book(pdf)
    try:
        yield book
    finally:
        pdf.close()


from src.reporting.figures.validate import (  # noqa: E402
    assert_no_default_mpl_colors,
)
