"""Always-on scan: production CSV readers must not point at Downloads or repo data/*.csv."""
from __future__ import annotations

import ast
from pathlib import Path

from mascotrl.data.paths import DATA_ROOT, LSEG_RAW, RAW_ROOT, TIER_A, TIER_A_DIR, TIER_B, UNIVERSE_IDENTIFIERS, MASCOTRL_ROOT

SKIP_DIRS = {
    ".venv",
    ".git",
    "archive",
    "authority",
    "attic",
    "graveyard",
        "logs",
    "node_modules",
}
SCAN_ROOTS = ("src", "scripts", "tests", "config")
ALLOW_FILES = {
    "tests/test_raw_path_literals.py",
    "scripts/organize_volsurf_raw.py",
}


def _iter_py() -> list[Path]:
    out: list[Path] = []
    for root in SCAN_ROOTS:
        base = MASCOTRL_ROOT / root
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            rel = path.relative_to(MASCOTRL_ROOT).as_posix()
            if rel in ALLOW_FILES:
                continue
            out.append(path)
    return out


def test_tier_paths_resolve_under_raw_root() -> None:
    assert "volsurf_raw" in str(RAW_ROOT) or Path(RAW_ROOT).name == "volsurf_raw"
    assert TIER_A_DIR == Path(RAW_ROOT) / "om"
    for key, path in TIER_A.items():
        resolved = Path(path)
        assert RAW_ROOT in resolved.parents or resolved.parent == RAW_ROOT / "om", key
        assert "Downloads" not in str(resolved)
    for key, path in TIER_B.items():
        resolved = Path(path)
        assert resolved.parent == Path(RAW_ROOT) / "macro", key
        assert MASCOTRL_ROOT / "data" not in resolved.parents
    assert UNIVERSE_IDENTIFIERS.parent == Path(RAW_ROOT) / "identifiers"
    assert LSEG_RAW == Path(RAW_ROOT) / "lseg"


def test_no_production_downloads_or_data_csv_literals() -> None:
    forbidden_abs = "/mnt/volsurf/Downloads"
    hits: list[str] = []
    for path in _iter_py():
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(MASCOTRL_ROOT).as_posix()
        if forbidden_abs in text:
            hits.append(f"{rel}: absolute Downloads path")
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                left = ast.unparse(node.left)
                right = ast.unparse(node.right)
                if "MOUNT_ROOT" in left and "Downloads" in right:
                    hits.append(f"{rel}: MOUNT_ROOT / Downloads")
                if "DATA_ROOT" in left and ".csv" in right:
                    hits.append(f"{rel}: DATA_ROOT / csv")
    assert hits == [], "forbidden raw-CSV literals:\n" + "\n".join(hits)


def test_no_vendor_csv_left_under_data_root() -> None:
    leftover = [
        p.relative_to(DATA_ROOT).as_posix()
        for p in DATA_ROOT.rglob("*.csv")
        if p.is_file() and "pseudo" not in p.parts and "lseg_data_updated" not in p.parts
    ]
    assert leftover == [], leftover
