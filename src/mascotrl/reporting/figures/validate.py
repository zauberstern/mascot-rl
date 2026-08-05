"""Figure figure validators (fatal in strict export).

Also hosts ``assert_no_default_mpl_colors`` (moved from ``book_style``; re-exported
there for ops-book callers).
"""
from __future__ import annotations

import re
import warnings
from typing import Any

from mascotrl.reporting.figures.labels import AXIS_LABELS, is_snake_case

_DEFAULT_MPL_CYCLE_HEXES = frozenset(
    c.lower()
    for c in (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    )
)

_SNAKE = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)+$")
_HAS_UNIT = re.compile(r"\([^)]+\)")


def _artist_colors(fig: Any) -> set[str]:
    import matplotlib.colors as mcolors

    colors: set[str] = set()
    for ax in fig.get_axes():
        for line in ax.get_lines():
            try:
                colors.add(mcolors.to_hex(line.get_color()).lower())
            except (ValueError, TypeError):
                continue
        for patch in ax.patches:
            try:
                colors.add(mcolors.to_hex(patch.get_facecolor()[:3]).lower())
            except (ValueError, TypeError, IndexError):
                continue
    return colors


def assert_no_default_mpl_colors(fig: Any, *, stem: str = "") -> None:
    """Hard rule: no default matplotlib tab10 colors in a shipped figure."""
    used = _artist_colors(fig)
    offending = used & _DEFAULT_MPL_CYCLE_HEXES
    if offending:
        prefix = f"{stem}: " if stem else ""
        raise AssertionError(
            f"{prefix}figure uses default matplotlib cycle colors: {sorted(offending)}; "
            "use FAMILY_PALETTE / SLEEVE_PALETTE / figure colour constants instead"
        )


def assert_axis_labels_human(fig: Any, *, stem: str = "") -> None:
    """Non-empty axis labels must be registered or carry a parenthetical unit."""
    allowed = set(AXIS_LABELS.values())
    prefix = f"{stem}: " if stem else ""
    for ax in fig.get_axes():
        for lab in (ax.get_xlabel(), ax.get_ylabel()):
            text = str(lab or "").strip()
            if not text:
                continue
            if text in allowed:
                continue
            if _HAS_UNIT.search(text):
                continue
            raise AssertionError(
                f"{prefix}axis label not human/units-registered: {text!r}"
            )


def assert_no_raw_identifiers(fig: Any, *, stem: str = "") -> None:
    """No tick label, legend entry, or annotation may be snake_case."""
    prefix = f"{stem}: " if stem else ""
    offenders: list[str] = []

    def _check(text: str, where: str) -> None:
        t = str(text or "").strip()
        if t and is_snake_case(t):
            offenders.append(f"{where}={t!r}")

    for ax in fig.get_axes():
        for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            _check(lab.get_text(), "tick")
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                _check(t.get_text(), "legend")
        for t in ax.texts:
            _check(t.get_text(), "annotation")
    for t in fig.texts:
        _check(t.get_text(), "figtext")
    if offenders:
        raise AssertionError(
            f"{prefix}raw snake_case identifier(s) in figure: {', '.join(offenders[:8])}"
        )


def _text_artists(fig: Any) -> list[tuple[Any, str, Any]]:
    """Return ``(artist, kind, axes_or_None)`` for text elements we validate."""
    arts: list[tuple[Any, str, Any]] = []
    for ax in fig.get_axes():
        if ax.get_title():
            arts.append((ax.title, "title", ax))
        if ax.xaxis.label.get_text():
            arts.append((ax.xaxis.label, "xlabel", ax))
        if ax.yaxis.label.get_text():
            arts.append((ax.yaxis.label, "ylabel", ax))
        for lab in ax.get_xticklabels():
            arts.append((lab, "xtick", ax))
        for lab in ax.get_yticklabels():
            arts.append((lab, "ytick", ax))
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                arts.append((t, "legend", ax))
        for t in ax.texts:
            arts.append((t, "annotation", ax))
    for t in fig.texts:
        arts.append((t, "figtext", None))
    return [
        (a, k, ax)
        for a, k, ax in arts
        if str(getattr(a, "get_text", lambda: "")()).strip()
    ]


def _extent(artist: Any, renderer: Any):
    try:
        return artist.get_window_extent(renderer=renderer)
    except Exception:
        try:
            return artist.get_window_extent(renderer)
        except Exception:
            return None


def _pair_exempt(kind_i: str, kind_j: str, ax_i: Any, ax_j: Any) -> bool:
    """Axis labels abut their ticks; tick AABBs abut neighbours — not defects."""
    if ax_i is not None and ax_i is ax_j:
        tickish = {"xtick", "ytick"}
        labelish = {"xlabel", "ylabel"}
        if kind_i in tickish and kind_j in tickish:
            return True
        if (kind_i in labelish and kind_j in tickish) or (
            kind_j in labelish and kind_i in tickish
        ):
            return True
        if kind_i in labelish and kind_j in labelish:
            return True
    # Small-multiple grids: neighbouring panel ticks/labels often share an edge.
    if ax_i is not None and ax_j is not None and ax_i is not ax_j:
        if kind_i in {"xtick", "ytick", "xlabel", "ylabel"} and kind_j in {
            "xtick",
            "ytick",
            "xlabel",
            "ylabel",
        }:
            return True
    return False


def assert_no_overlap(fig: Any, *, stem: str = "", tol_px: float = 1.0) -> None:
    """After draw, non-exempt text window extents must not meaningfully intersect."""
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    prefix = f"{stem}: " if stem else ""
    extents = []
    for a, kind, ax in _text_artists(fig):
        ext = _extent(a, renderer)
        if ext is None or ext.width < 0.5 or ext.height < 0.5:
            continue
        extents.append((a, kind, ax, ext, str(a.get_text())[:60]))

    for i in range(len(extents)):
        _a_i, k_i, ax_i, e_i, t_i = extents[i]
        for j in range(i + 1, len(extents)):
            _a_j, k_j, ax_j, e_j, t_j = extents[j]
            if _pair_exempt(k_i, k_j, ax_i, ax_j):
                continue
            inter = Bbox.intersection(e_i, e_j)
            if inter is None:
                continue
            if inter.width > tol_px and inter.height > tol_px:
                area = float(inter.width * inter.height)
                if area <= tol_px * tol_px:
                    continue
                raise AssertionError(
                    f"{prefix}overlapping text artists: {t_i!r} vs {t_j!r}"
                )


def assert_within_canvas(fig: Any, *, stem: str = "", pad_px: float = 8.0) -> None:
    """Visible text must not be clipped off the figure canvas.

    Axis-label AABBs overhang after rotation; those use a centre check.
    Tick labels outside the axes view interval are ignored (mpl keeps extras).
    Other text uses full-extent containment.
    """
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    prefix = f"{stem}: " if stem else ""
    cw, ch = fig.canvas.get_width_height()
    fb = Bbox.from_bounds(0.0, 0.0, float(cw), float(ch))

    def _outside(ext: Any, *, centre: bool) -> bool:
        if centre:
            cx = 0.5 * (ext.x0 + ext.x1)
            cy = 0.5 * (ext.y0 + ext.y1)
            return (
                cx < fb.x0 - pad_px
                or cy < fb.y0 - pad_px
                or cx > fb.x1 + pad_px
                or cy > fb.y1 + pad_px
            )
        return (
            ext.x0 < fb.x0 - pad_px
            or ext.y0 < fb.y0 - pad_px
            or ext.x1 > fb.x1 + pad_px
            or ext.y1 > fb.y1 + pad_px
        )

    for a, kind, ax in _text_artists(fig):
        ext = _extent(a, renderer)
        if ext is None or ext.width < 0.5 or ext.height < 0.5:
            continue
        if kind in ("xtick", "ytick") and ax is not None:
            # Drop major labels for ticks outside the current view.
            try:
                x, y = a.get_position()
                if kind == "xtick":
                    x0, x1 = ax.get_xlim()
                    lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
                    if not (lo - 1e-9 <= float(x) <= hi + 1e-9):
                        continue
                else:
                    y0, y1 = ax.get_ylim()
                    lo, hi = (y0, y1) if y0 <= y1 else (y1, y0)
                    if not (lo - 1e-9 <= float(y) <= hi + 1e-9):
                        continue
            except Exception:
                pass
        use_centre = kind in {"xlabel", "ylabel", "title"}
        if _outside(ext, centre=use_centre):
            raise AssertionError(
                f"{prefix}text clipped outside canvas: {a.get_text()!r} ({kind})"
            )


def assert_legend_present(fig: Any, *, stem: str = "") -> None:
    """Axes with more than one labelled artist must have a legend."""
    prefix = f"{stem}: " if stem else ""
    for ax in fig.get_axes():
        labelled = 0
        for line in ax.get_lines():
            if line.get_label() and not str(line.get_label()).startswith("_"):
                labelled += 1
        for patch in ax.patches:
            lab = patch.get_label() if hasattr(patch, "get_label") else ""
            if lab and not str(lab).startswith("_"):
                labelled += 1
        for coll in ax.collections:
            lab = coll.get_label() if hasattr(coll, "get_label") else ""
            if lab and not str(lab).startswith("_"):
                labelled += 1
        if labelled > 1 and ax.get_legend() is None and fig.legends == []:
            raise AssertionError(
                f"{prefix}axes has {labelled} labelled artists but no legend"
            )


def assert_greyscale_separable(fig: Any, *, stem: str = "") -> None:
    """Axes with more than three labelled line series must use >=3 line styles."""
    prefix = f"{stem}: " if stem else ""
    for ax in fig.get_axes():
        lines = []
        for ln in ax.get_lines():
            if not ln.get_visible():
                continue
            lab = str(ln.get_label() or "")
            if not lab or lab.startswith("_"):
                continue
            ls = str(ln.get_linestyle())
            if ls in ("None", "none", " ", ""):
                continue
            if len(ln.get_xdata()) == 0:
                continue
            lines.append(ln)
        if len(lines) <= 3:
            continue
        styles = {str(ln.get_linestyle()) for ln in lines}
        normalised = set()
        for s in styles:
            if s in ("-", "solid"):
                normalised.add("solid")
            elif s in ("--", "dashed"):
                normalised.add("dashed")
            elif s in ("-.", "dashdot"):
                normalised.add("dashdot")
            elif s in (":", "dotted"):
                normalised.add("dotted")
            else:
                normalised.add(s)
        if len(normalised) < 3:
            raise AssertionError(
                f"{prefix}axes has {len(lines)} lines but only {len(normalised)} "
                f"distinct styles {sorted(normalised)}; need >= 3 for greyscale safety"
            )


def assert_tick_budget(
    ax_or_fig: Any, *, max_major: int = 18, stem: str = "", strict: bool = True
) -> None:
    """Fail when a 2D axes has more than max_major ticks on x or y."""
    axes = ax_or_fig.get_axes() if hasattr(ax_or_fig, "get_axes") else [ax_or_fig]
    prefix = f"{stem}: " if stem else ""
    offenders: list[str] = []

    def _count(ax: Any, axis: str) -> int:
        ticks = ax.get_xticks() if axis == "x" else ax.get_yticks()
        labels = (
            ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
        )
        n = 0
        for tick, lab in zip(ticks, labels):
            if not lab.get_visible():
                continue
            if str(lab.get_text() or "").strip() or tick == 0:
                n += 1
        return n

    for ax in axes:
        if getattr(ax, "name", None) == "3d":
            continue
        for axis in ("x", "y"):
            count = _count(ax, axis)
            if count > max_major:
                offenders.append(f"{axis}={count}")

    if offenders:
        msg = f"{prefix}tick budget exceeded (max {max_major}): {', '.join(offenders)}"
        if strict:
            raise AssertionError(msg)
        warnings.warn(f"figure validator (non-strict): {msg}", stacklevel=2)


def run_figure_validators(fig: Any, *, stem: str = "", strict: bool = True) -> None:
    """Run all seven validators; raise in strict mode, warn otherwise."""
    has_3d = any(getattr(ax, "name", None) == "3d" for ax in fig.get_axes())
    checks = [
        assert_no_default_mpl_colors,
        assert_axis_labels_human,
        assert_no_raw_identifiers,
        assert_legend_present,
        assert_greyscale_separable,
    ]
    # 3D projection tick AABBs routinely fall outside the 2D canvas; layout
    # validators apply to planar figures (B5b carries readable coords).
    if not has_3d:
        checks.extend([assert_no_overlap, assert_within_canvas, assert_tick_budget])
    errors: list[str] = []
    for fn in checks:
        try:
            fn(fig, stem=stem)
        except AssertionError as exc:
            if strict:
                raise
            errors.append(str(exc))
    for msg in errors:
        warnings.warn(f"figure validator (non-strict): {msg}", stacklevel=2)
