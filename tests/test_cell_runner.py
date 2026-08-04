"""Cell runner correctness: OOB exit, extract safety, artifact gates."""
from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from deploy.aws_burst.docker import cell_runner


def _base_env(manifest_uri: str = "s3://b/m.json") -> dict[str, str]:
    return {
        "MASCOTRL_WAVE": "PICK",
        "MASCOTRL_SHARD_MANIFEST_URI": manifest_uri,
        "MASCOTRL_PANEL_URI": "s3://b/panel_bundle.tar",
        "MASCOTRL_PANEL_SHA256": "deadbeef",
        "MASCOTRL_OUT_URI": "s3://b/PICK/",
        "MASCOTRL_CONTAINER_DIGEST": "sha256:abc",
        "MASCOTRL_COMPUTE_HOST": "remote",
        "AWS_BATCH_JOB_ARRAY_INDEX": "0",
    }


def test_required_env_missing_exits() -> None:
    with pytest.raises(SystemExit, match="cell_runner_env_missing"):
        cell_runner.main()


def test_array_index_defaults_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _base_env()
    env.pop("AWS_BATCH_JOB_ARRAY_INDEX")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("AWS_BATCH_JOB_ARRAY_INDEX", raising=False)
    monkeypatch.delenv("MASCOTRL_ARRAY_INDEX", raising=False)
    got = cell_runner._required_env()
    assert got["AWS_BATCH_JOB_ARRAY_INDEX"] == "0"


def test_array_index_out_of_range_exits_3(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _base_env()
    env["AWS_BATCH_JOB_ARRAY_INDEX"] = "5"
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    s3 = MagicMock()
    s3.get_object.return_value = {
        "Body": MagicMock(read=lambda: json.dumps({"cells": ["a.yaml"]}).encode())
    }
    monkeypatch.setattr(cell_runner, "_s3_client", lambda: s3)
    assert cell_runner.main() == 3


def test_safe_extractall_blocks_traversal(tmp_path: Path) -> None:
    tar_path = tmp_path / "evil.tar"
    with tarfile.open(tar_path, "w") as tar:
        info = tarfile.TarInfo(name="../escape.txt")
        data = b"x"
        info.size = len(data)
        import io

        tar.addfile(info, io.BytesIO(data))
    dest = tmp_path / "out"
    dest.mkdir()
    with tarfile.open(tar_path, "r") as tar:
        # Python 3.12+ data_filter also blocks; either path is fine.
        with pytest.raises((RuntimeError, tarfile.OutsideDestinationError, Exception)):
            cell_runner._safe_extractall(tar, dest)


def test_upload_error_best_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _base_env()
    put = MagicMock()
    s3 = MagicMock()
    s3.put_object = put
    monkeypatch.setattr(cell_runner, "_s3_client", lambda: s3)
    cell_runner._upload_error(env, "cell_a", RuntimeError("boom"))
    assert put.called
    key = put.call_args.kwargs["Key"]
    assert key.endswith("cell_a.error.json")
