"""Shared figure/table primitives for the equity allocation book."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from mascotrl.reporting.book_style import (
    C_ACCENT,
    C_GRAY,
    C_NAVY,
    C_NEG,
    C_POS,
    C_STEEL,
    C_ZERO,
    CMAP_DIVERGING,
    HEIGHT_DEFAULT,
    HEIGHT_SINGLE,
    WIDTH_DOUBLE_IN,
    WIDTH_SINGLE_IN,
    PdfBook,
    caption,
    family_color,
    finalize_figure,
    place_legend,
    stamp_n,
    table_figure,
)
from mascotrl.reporting.book_style import save_pdf_png, use_agg

def _new_fig(
    *,
    width: str = "double",
    height_in: float | None = None,
    legend_space: bool = False,
    nrows: int = 1,
    ncols: int = 1,
    **kw: Any,
):
    import matplotlib.pyplot as plt

    use_agg()
    w = WIDTH_DOUBLE_IN if width == "double" else WIDTH_SINGLE_IN
    if legend_space:
        w += 1.6
    h = height_in if height_in is not None else (
        HEIGHT_SINGLE if width == "single" else HEIGHT_DEFAULT
    )
    h = float(min(max(h, 2.5), 9.5))
    return plt.subplots(nrows=nrows, ncols=ncols, figsize=(w, h), **kw)


def _finish(
    fig,
    *,
    stem: Path,
    manifest: Mapping[str, Any],
    book: Any | None,
    legend_space: bool = False,
    caption_text: str | None = None,
) -> dict[str, str]:
    """Stamp footer + optional caption, save PNG (book.pdf via PdfBook), close."""
    import matplotlib.pyplot as plt

    finalize_figure(fig, legend_space=legend_space)
    if caption_text:
        caption(fig, caption_text)
    from mascotrl.reporting import eq_alloc_book as _book

    _book.stamp_footer(fig, manifest)
    paths = save_pdf_png(fig, stem, pdf=False)
    if book is not None:
        book.add(fig)
    plt.close(fig)
    return paths


def _entry(
    fig_id: str,
    title: str,
    *,
    status: str,
    paths: dict[str, str] | None = None,
    sources: list[str] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": fig_id,
        "title": title,
        "status": status,
        "sources": sources or [],
    }
    if paths:
        out.update(paths)
    if note:
        out["note"] = note
    return out


def _write_table(df: pd.DataFrame, out_dir: Path, name: str) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    json_path = out_dir / f"{name}.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2))
    return {"csv": str(csv_path), "json": str(json_path)}


def _finite(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _annualized_sharpe(r: np.ndarray, periods: int = 252) -> float:
    from mascotrl.eval.stats_rigor import annualized_sharpe

    return float(annualized_sharpe(np.asarray(r, dtype=np.float64)))


def _focus_frame(
    strategy_frames: Mapping[str, pd.DataFrame], focus: str
) -> tuple[str, pd.DataFrame] | tuple[None, None]:
    if focus in strategy_frames and not strategy_frames[focus].empty:
        return focus, strategy_frames[focus]
    for name, df in strategy_frames.items():
        if df is not None and not df.empty:
            return name, df
    return None, None


def _weight_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("w_")]


# --------------------------------------------------------------------------
# Section 0: provenance
# --------------------------------------------------------------------------


