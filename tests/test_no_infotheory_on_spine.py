"""CRUCIBLE Part B: equity spine must not import or reference infotheory."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (ROOT / "src", ROOT / "scripts", ROOT / "config")
BANNED_SELECTORS = ("select_universe_dii", "select_universe_te")


def _iter_text_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        if path.suffix.lower() in {".pyc", ".pyo", ".so", ".png", ".jpg", ".parquet", ".npz"}:
            continue
        yield path


def test_no_infotheory_string_under_src_scripts_config() -> None:
    hits: list[str] = []
    for root in SCAN_ROOTS:
        if not root.is_dir():
            continue
        for path in _iter_text_files(root):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if "infotheory" in text:
                hits.append(str(path.relative_to(ROOT)))
    assert hits == [], "infotheory still referenced in:\n" + "\n".join(hits)


def test_no_dii_te_selector_calls_under_src_scripts() -> None:
    hits: list[str] = []
    for root in (ROOT / "src", ROOT / "scripts"):
        for path in _iter_text_files(root):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for banned in BANNED_SELECTORS:
                if banned in text:
                    hits.append(f"{path.relative_to(ROOT)}:{banned}")
    assert hits == [], "banned selectors still present:\n" + "\n".join(hits)


def test_densest_subgraph_greedy_lives_in_graph_helpers() -> None:
    from src.reporting.figures.graph_helpers import densest_subgraph_greedy

    assert callable(densest_subgraph_greedy)


def test_src_data_does_not_import_graph_helpers() -> None:
    """densest-subgraph helper is plot/diagnostic only; data layer must not select with it."""
    data_root = ROOT / "src" / "data"
    hits: list[str] = []
    for path in _iter_text_files(data_root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "graph_helpers" in text or "densest_subgraph_greedy" in text:
            hits.append(str(path.relative_to(ROOT)))
    assert hits == [], "src/data must not import densest helpers:\n" + "\n".join(hits)
