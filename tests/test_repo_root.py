"""Repository root resolution after mascotrl package move."""
from __future__ import annotations

from pathlib import Path

from mascotrl._root import REPO_ROOT


def test_repo_root_points_at_extract_root() -> None:
    assert REPO_ROOT == Path(__file__).resolve().parents[1]
    assert (REPO_ROOT / "pyproject.toml").is_file()
    assert (REPO_ROOT / "src" / "mascotrl" / "data" / "slot_mask.py").is_file()
