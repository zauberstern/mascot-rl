"""Only approved markdown files may exist in the public extract."""
from __future__ import annotations

from pathlib import Path

ALLOWED_ROOT_MD = frozenset(
    {
        "README.md",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
    }
)


def test_markdown_allowlist() -> None:
    root = Path(__file__).resolve().parents[1]
    unexpected: list[str] = []
    for path in root.rglob("*.md"):
        if any(part in {".git", ".venv", "build", "logs", ".pytest_cache"} for part in path.parts):
            continue
        rel = path.relative_to(root).as_posix()
        if rel.startswith(".github/"):
            continue
        if rel not in ALLOWED_ROOT_MD:
            unexpected.append(rel)
    assert unexpected == [], "unexpected markdown files:\n" + "\n".join(sorted(unexpected))
