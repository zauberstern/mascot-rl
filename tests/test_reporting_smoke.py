"""Reporting stack smoke: matplotlib Agg + book_style helpers."""
from __future__ import annotations

from pathlib import Path

import pytest


def test_book_style_academic_figure_smoke(tmp_path: Path):
    from src.reporting.book_style import academic_figure, apply_academic_rc, save_figure, use_agg

    use_agg()
    apply_academic_rc()
    with academic_figure() as (fig, ax):
        ax.plot([0, 1, 2], [0.0, 0.1, -0.05])
        ax.set_xlabel("t (days)")
        ax.set_ylabel("pnl")
        out = tmp_path / "smoke.png"
        paths = save_figure(fig, out, dpi=72)
        assert any(Path(p).exists() for p in paths)


def test_seaborn_optional_import_does_not_break_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    plt.close(fig)
    seaborn = pytest.importorskip("seaborn")
    assert seaborn is not None
