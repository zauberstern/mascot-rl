"""Remote provenance stamping in spectrum campaign."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.run_spectrum_campaign import run_cell


ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT / "config/spectrum/cherrypick/eq_K100_single_ppo_mlp_softmax_mean_std_cao.yaml"


@pytest.fixture
def remote_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    lock = tmp_path / "requirements.lock"
    lock.write_text("pytest\n", encoding="utf-8")
    monkeypatch.setenv("MASCOTRL_COMPUTE_HOST", "remote")
    monkeypatch.setenv("MASCOTRL_CONTAINER_DIGEST", "sha256:deadbeef")
    monkeypatch.setenv("MASCOTRL_REQUIREMENTS_LOCK", str(lock))


def test_remote_dry_run_stamps_provenance(remote_env: None) -> None:
    with patch(
        "src.reporting.provenance_stamp.fetch_imds_instance_type",
        return_value="c5.large",
    ):
        art = run_cell(CELL, dry_run=True)
    assert art.get("compute_host") == "remote"
    assert art.get("container_digest") == "sha256:deadbeef"
    assert art.get("instance_type") == "c5.large"
    assert art.get("requirements_lock_sha256")
