"""Manifest builders for AWS burst wave submits."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def build_manifest(
    *,
    wave: str,
    region: str,
    cells: list[str],
    shards: list[dict[str, Any]],
    local_equivalent: str,
) -> dict[str, Any]:
    return {
        "wave": wave,
        "region": region,
        "n_cells": len(cells),
        "cells": list(cells),
        "shards": list(shards),
        "local_equivalent": local_equivalent,
    }


def manifest_sha256(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
