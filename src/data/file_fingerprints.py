"""File fingerprint helpers for copy-verify workflows."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

CHUNK = 1024 * 1024


def header_sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.sha256(f.readline()).hexdigest()


def _sha_range(path: Path, *, start: int, length: int) -> str:
    h = hashlib.sha256()
    size = path.stat().st_size
    if size <= 0 or length <= 0:
        return h.hexdigest()
    start = max(0, min(start, size))
    length = min(length, size - start)
    with path.open("rb") as f:
        f.seek(start)
        remaining = length
        while remaining > 0:
            buf = f.read(min(CHUNK, remaining))
            if not buf:
                break
            h.update(buf)
            remaining -= len(buf)
    return h.hexdigest()


def file_fingerprints(path: Path) -> dict[str, Any]:
    size = int(path.stat().st_size)
    tail_start = max(0, size - CHUNK)
    mid_start = max(0, (size - min(CHUNK, size)) // 2)
    return {
        "size": size,
        "header_sha256": header_sha256(path),
        "head_1mib_sha256": _sha_range(path, start=0, length=min(CHUNK, size)),
        "mid_1mib_sha256": _sha_range(path, start=mid_start, length=min(CHUNK, size)),
        "tail_1mib_sha256": _sha_range(path, start=tail_start, length=min(CHUNK, size)),
    }


def fingerprints_match(live: dict[str, Any], stored: dict[str, Any] | None) -> bool:
    """Compare size/header/head/tail. Mid is optional for older MANIFEST rows."""
    if not stored:
        return False
    keys = ("size", "header_sha256", "head_1mib_sha256", "tail_1mib_sha256")
    if any(live.get(k) != stored.get(k) for k in keys):
        return False
    if "mid_1mib_sha256" in stored and live.get("mid_1mib_sha256") != stored.get("mid_1mib_sha256"):
        return False
    return True
