"""Tests for supervised long-job runner + job memory sanity."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mascotrl.aws_burst.jobdef import (
    assert_job_memory_fits_instance,
    build_job_definition,
)


ROOT = Path(__file__).resolve().parents[1]


def test_assert_job_memory_fits_m7i_flex_large() -> None:
    assert_job_memory_fits_instance(4096)  # CFN default
    with pytest.raises(ValueError, match="job_memory_exceeds_instance"):
        assert_job_memory_fits_instance(7000)


def test_build_job_definition_refuses_oversized_memory() -> None:
    with pytest.raises(ValueError, match="job_memory_exceeds_instance"):
        build_job_definition(
            image_uri="example.com/img@sha256:dead",
            job_role_arn="arn:aws:iam::1:role/x",
            memory_mb=7000,
        )


def test_run_supervised_records_exit(tmp_path: Path) -> None:
    status = tmp_path / "job.status.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_supervised.py"),
            "--status",
            str(status),
            "--heartbeat-seconds",
            "0.2",
            "--",
            sys.executable,
            "-c",
            "print('ok'); raise SystemExit(0)",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert status.is_file()
    payload = json.loads(status.read_text(encoding="utf-8"))
    assert payload["state"] == "exited"
    assert payload["exit_code"] == 0
    assert payload["pid"] > 0


def test_run_supervised_captures_nonzero(tmp_path: Path) -> None:
    status = tmp_path / "fail.status.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_supervised.py"),
            "--status",
            str(status),
            "--heartbeat-seconds",
            "0.2",
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(7)",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 7
    payload = json.loads(status.read_text(encoding="utf-8"))
    assert payload["exit_code"] == 7
