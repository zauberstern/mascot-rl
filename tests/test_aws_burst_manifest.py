"""Manifest hashing stability."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.plumbing

from mascotrl.aws_burst.manifest import build_manifest, manifest_sha256


def test_manifest_sha_stable() -> None:
    m = build_manifest(
        wave="PICK",
        region="eu-central-1",
        cells=["a.yaml", "b.yaml"],
        shards=[{"profile": "volsurf-burst-1", "shard": 0}],
        local_equivalent="echo",
    )
    h1 = manifest_sha256(m)
    h2 = manifest_sha256(m)
    assert h1 == h2
    assert len(h1) == 64
