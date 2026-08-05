"""Moto-backed submit/pull roundtrips for AWS burst."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import boto3
import pytest

pytestmark = pytest.mark.plumbing
from moto import mock_aws

from scripts import aws_pull_artifacts as pull_mod
from scripts import aws_submit_wave as submit_mod
from mascotrl.aws_burst.aws_client import BurstClient
from mascotrl.aws_burst.profiles import MAX_VCPUS_PER_ACCOUNT


ROOT = Path(__file__).resolve().parents[1]


def test_pick_dry_run_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        submit_mod,
        "armed_profiles",
        lambda _root: [
            {"profile": "volsurf-burst-1", "shard": "0", "account_id": "1"},
            {"profile": "volsurf-burst-2", "shard": "1", "account_id": "2"},
            {"profile": "volsurf-burst-3", "shard": "2", "account_id": "3"},
        ],
    )
    monkeypatch.setattr(submit_mod, "check_submit_allowed", lambda **kwargs: None)
    out = submit_mod.submit_wave(ROOT, "PICK_SMOKE", dry_run=True)
    assert out["dry_run"] is True
    assert out["plan"]["manifest"]["n_cells"] == 1
    assert "expected_cells" in out["plan"]


@mock_aws
def test_s3_put_get_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-1")

    real_session = boto3.Session

    def _session(profile_name=None, region_name=None):
        return real_session(region_name=region_name or "eu-central-1")

    monkeypatch.setattr("mascotrl.aws_burst.aws_client.boto3.Session", _session)
    client = BurstClient("volsurf-burst-1", "eu-central-1")
    bucket = "volsurf-burst-1-artifacts"
    client.ensure_bucket(bucket)
    art = tmp_path / "cell.json"
    art.write_text('{"ok": true, "compute_host": "remote"}\n', encoding="utf-8")
    digest = client.put_file_with_sha(bucket, "PICK/cell.json", art)
    assert len(digest) == 64
    keys = client.list_keys(bucket, "PICK/")
    assert "PICK/cell.json" in keys
    assert "PICK/cell.json.sha256" in keys
    got = client.get_json(bucket, "PICK/cell.json")
    assert got["ok"] is True


def test_pull_completeness_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wave = "PICK_SMOKE"
    manifest = {
        "wave": wave,
        "expected_cells": ["cell_a", "cell_b"],
        "n_expected": 2,
        "cells": ["config/a/cell_a.yaml", "config/a/cell_b.yaml"],
    }
    cfg = tmp_path / "deploy/aws_burst/config"
    cfg.mkdir(parents=True)
    (cfg / f"wave_{wave}_manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    for p in ("volsurf-burst-1", "volsurf-burst-2", "volsurf-burst-3"):
        (cfg / f"budget_armed_{p}.json").write_text(
            '{"verified": true, "action_id": "a", "armed": true}\n',
            encoding="utf-8",
        )

    monkeypatch.setattr(
        pull_mod,
        "armed_profiles",
        lambda _root: [
            {"profile": "volsurf-burst-1", "account_id": "1"},
            {"profile": "volsurf-burst-2", "account_id": "2"},
            {"profile": "volsurf-burst-3", "account_id": "3"},
        ],
    )

    class _NoopClient:
        def __init__(self, *_a, **_k):
            pass

        def account_id(self):
            return "1"

        def download_prefix(self, *_a, **_k):
            return []

    monkeypatch.setattr(pull_mod, "BurstClient", _NoopClient)
    monkeypatch.setattr(pull_mod, "artifact_bucket", lambda _a: "b")

    with pytest.raises(SystemExit, match="pull_incomplete"):
        pull_mod.pull_wave(tmp_path, wave, dest=tmp_path / "out", require_complete=True)


def test_deploy_script_default_maxvcpus() -> None:
    text = (ROOT / "deploy/aws_burst/scripts/aws_deploy_batch.sh").read_text(encoding="utf-8")
    assert 'MAXV="${1:-32}"' in text
    assert MAX_VCPUS_PER_ACCOUNT == 32
