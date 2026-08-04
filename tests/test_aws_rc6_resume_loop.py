"""Unit tests for RC6 resume-loop helpers (no live AWS)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.plumbing

from scripts import aws_rc6_resume_loop as loop

ROOT = Path(__file__).resolve().parents[1]


def test_count_active_wave_parents_skips_children(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        loop,
        "armed_profiles",
        lambda _root: [{"profile": "volsurf-burst-1", "account_id": "1"}],
    )

    class _Client:
        profile = "volsurf-burst-1"

        def list_jobs(self, _queue: str, status: str):
            if status != "RUNNING":
                return []
            return [
                {"jobId": "parent-1", "jobName": "mascotrl-RC6_NARRATIVE-abc-himem"},
                {"jobId": "parent-1:0", "jobName": "mascotrl-RC6_NARRATIVE-abc-himem"},
                {"jobId": "other", "jobName": "mascotrl-RC6-xyz-himem56"},
            ]

    monkeypatch.setattr(loop, "BurstClient", lambda *_a, **_k: _Client())
    assert loop.count_active_wave_parents(ROOT, "RC6_NARRATIVE") == 1
    assert loop.count_active_wave_parents(ROOT, "RC6") == 1
    assert loop.count_active_wave_parents(ROOT, "RC6_HAPPO") == 0
