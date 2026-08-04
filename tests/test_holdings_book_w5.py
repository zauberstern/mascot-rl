"""W5 holdings book + PNG-only figure save."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from src.reporting.book_style import save_pdf_png, use_agg
from src.reporting.book_style import FAMILY_ORDER
from src.reporting.strategy_persistence import write_holdings_book

def test_save_pdf_png_defaults_to_png_only(tmp_path: Path):
    use_agg()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = save_pdf_png(fig, tmp_path / 'fig')
    plt.close(fig)
    assert 'png' in out
    assert Path(out['png']).is_file()
    assert 'pdf' not in out
    assert not (tmp_path / 'fig.pdf').exists()
