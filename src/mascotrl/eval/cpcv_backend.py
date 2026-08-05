"""Resolve CPCV backend: purgedcv default when importable."""
from __future__ import annotations

from typing import Any, Mapping


def purgedcv_available() -> bool:
    try:
        import purgedcv  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_use_purgedcv(cfg: Mapping[str, Any] | None = None) -> bool:
    """Default True when purgedcv is installed; YAML ``use_purgedcv=false`` escapes."""
    cfg = cfg or {}
    if "use_purgedcv" in cfg:
        return bool(cfg.get("use_purgedcv"))
    return purgedcv_available()
