"""Per-profile account shape loading."""
from __future__ import annotations

from pathlib import Path

from src.aws_burst.profiles import (
    DEFAULT_ACCOUNT_SHAPE,
    MAX_VCPUS_PER_ACCOUNT,
    allowed_instance_types_for_profile,
    max_vcpus_for_profile,
    profile_shape,
)

ROOT = Path(__file__).resolve().parents[1]


def test_free_tier_default_on_unlisted_profile() -> None:
    # Unlisted profile name should fall back to free-tier defaults.
    assert max_vcpus_for_profile(ROOT, "volsurf-burst-unlisted") == MAX_VCPUS_PER_ACCOUNT
    assert allowed_instance_types_for_profile(ROOT, "volsurf-burst-unlisted") == frozenset(
        {"m7i-flex.large"}
    )
    shape = profile_shape(ROOT, "volsurf-burst-unlisted")
    assert shape["job_memory_mib"] == DEFAULT_ACCOUNT_SHAPE["job_memory_mib"]


def test_interim_paid_shape_burst3() -> None:
    assert max_vcpus_for_profile(ROOT, "volsurf-burst-3") == 300


def test_interim_paid_shape_burst1() -> None:
    assert max_vcpus_for_profile(ROOT, "volsurf-burst-1") == 500
    assert allowed_instance_types_for_profile(ROOT, "volsurf-burst-1") == frozenset(
        {"m7i", "c7i", "r7i"}
    )
