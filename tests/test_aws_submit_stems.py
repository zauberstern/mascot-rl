"""Stem-filter / spend-gate / attempt-timeout helpers for aws_submit_wave."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.plumbing

from scripts import aws_submit_wave as submit_mod
from src.aws_burst.job_routing import (
    JOB_DEFINITION_HIMEM56,
    himem_job_definition_for_memory,
)

ROOT = Path(__file__).resolve().parents[1]


def test_build_plan_stems_filter_keeps_four_shards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        submit_mod,
        "armed_profiles",
        lambda _root: [
            {"profile": "volsurf-burst-1", "shard": "0", "account_id": "1"},
            {"profile": "volsurf-burst-2", "shard": "1", "account_id": "2"},
            {"profile": "volsurf-burst-3", "shard": "2", "account_id": "3"},
            {"profile": "volsurf-burst-4", "shard": "3", "account_id": "4"},
        ],
    )
    stem = "eq_K100_single_ppo_mlp_sparse_tilt_differential_sharpe"
    plan = submit_mod.build_plan(
        ROOT,
        "RC6_NARRATIVE",
        stems=[stem],
    )
    assert plan["expected_cells"] == [stem]
    assert plan["n_shards"] == 4
    flat = [c for shard in plan["shard_plans"] for c in shard]
    assert len(flat) == 1
    assert Path(flat[0]).stem == stem
    assert sum(1 for s in plan["shard_plans"] if s) == 1


def test_build_plan_stems_unknown_raises() -> None:
    with pytest.raises(ValueError, match="stems_not_in_wave"):
        submit_mod.build_plan(
            ROOT,
            "RC6_NARRATIVE",
            stems=["does_not_exist_stem"],
        )


def test_remaining_wall_attempt_seconds_caps_and_buffers() -> None:
    deadline = datetime(2026, 8, 31, 16, 0, 0, tzinfo=timezone.utc)
    now = datetime(2026, 8, 30, 18, 40, 0, tzinfo=timezone.utc)
    secs = submit_mod.remaining_wall_attempt_seconds(
        now=now, deadline_utc=deadline, buffer_seconds=1200
    )
    # 21h20m wall - 20m buffer = 21h = 75600
    assert secs == 75600
    # Never invent a 48h timeout.
    assert secs <= 21 * 3600


def test_remaining_wall_attempt_seconds_past_deadline() -> None:
    deadline = datetime(2026, 8, 31, 16, 0, 0, tzinfo=timezone.utc)
    now = datetime(2026, 8, 31, 17, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="deadline_passed"):
        submit_mod.remaining_wall_attempt_seconds(now=now, deadline_utc=deadline)


def test_skip_spend_gate_skips_check_submit_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        submit_mod,
        "armed_profiles",
        lambda _root: [
            {"profile": "volsurf-burst-1", "shard": "0", "account_id": "1"},
            {"profile": "volsurf-burst-2", "shard": "1", "account_id": "2"},
            {"profile": "volsurf-burst-3", "shard": "2", "account_id": "3"},
            {"profile": "volsurf-burst-4", "shard": "3", "account_id": "4"},
        ],
    )
    called = {"n": 0}

    def _boom(**_kwargs):
        called["n"] += 1
        raise AssertionError("check_submit_allowed must not run")

    monkeypatch.setattr(submit_mod, "check_submit_allowed", _boom)
    out = submit_mod.submit_wave(
        ROOT,
        "RC6_NARRATIVE",
        dry_run=True,
        skip_spend_gate=True,
        stems=["eq_K100_single_ppo_mlp_sparse_tilt_differential_sharpe"],
    )
    assert out["dry_run"] is True
    assert called["n"] == 0
    assert out["plan"]["manifest"]["n_cells"] == 1


def test_job_definition_with_revision_suffix() -> None:
    assert (
        submit_mod.job_definition_with_revision("volsurf-burst-cell-himem", 28)
        == "volsurf-burst-cell-himem:28"
    )
    assert (
        submit_mod.job_definition_with_revision("volsurf-burst-cell-himem", None)
        == "volsurf-burst-cell-himem"
    )


def test_himem112_routes_to_himem56_jd() -> None:
    assert himem_job_definition_for_memory(114688) == JOB_DEFINITION_HIMEM56


def test_out_uri_prefix_equals_wave_name() -> None:
    """Stem-filtered submit must keep MASCOTRL_OUT_URI on the real wave."""
    from src.aws_burst.jobdef import build_container_env

    env = build_container_env(
        {
            "MASCOTRL_WAVE": "RC6_NARRATIVE",
            "MASCOTRL_SHARD_MANIFEST_URI": "s3://p/m.json",
            "MASCOTRL_PANEL_URI": "s3://p/panel_bundle.tar",
            "MASCOTRL_PANEL_SHA256": "abc",
            "MASCOTRL_OUT_URI": "s3://a/RC6_NARRATIVE/",
            "MASCOTRL_CONTAINER_DIGEST": "sha256:dead",
            "MASCOTRL_COMPUTE_HOST": "remote",
        }
    )
    by_name = {e["name"]: e["value"] for e in env}
    assert by_name["MASCOTRL_OUT_URI"] == "s3://a/RC6_NARRATIVE/"
    assert by_name["MASCOTRL_WAVE"] == "RC6_NARRATIVE"
