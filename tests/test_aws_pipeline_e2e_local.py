"""End-to-end local fake-remote path (no AWS credentials)."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.plumbing

import json
from pathlib import Path

from src.aws_burst.manifest import build_manifest, manifest_sha256
from src.aws_burst.waves import discover_wave_cells


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_to_sha_roundtrip() -> None:
    cells = discover_wave_cells(ROOT, "PICK_SMOKE")
    m = build_manifest(
        wave="PICK_SMOKE",
        region="eu-central-1",
        cells=cells,
        shards=[{"profile": "volsurf-burst-1", "shard": 0}],
        local_equivalent="echo",
    )
    h = manifest_sha256(m)
    assert len(h) == 64
    assert m["n_cells"] == 1
    assert json.loads(json.dumps(m)) == m
