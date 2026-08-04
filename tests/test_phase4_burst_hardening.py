"""Phase 4 AWS Burst hardening: digest pins, gates, pull, watcher (unit)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.plumbing

ROOT = Path(__file__).resolve().parents[1]


def _write_digest(cfg: Path, profile: str, digest: str, acct: str = "111") -> Path:
    path = cfg / f"image_digest_{profile}.json"
    path.write_text(
        json.dumps(
            {
                "profile": profile,
                "image": f"{acct}.dkr.ecr.eu-central-1.amazonaws.com/volsurf-burst:latest",
                "digest": digest,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _armed_payload(action_id: str = "act-1") -> str:
    return json.dumps(
        {"verified": True, "action_id": action_id, "armed": True, "budget_action_armed": True}
    ) + "\n"


# ---------------------------------------------------------------------------
# 31. CFN / smoke digest pin
# ---------------------------------------------------------------------------


def test_pinned_image_uri_uses_per_profile_digest(tmp_path: Path) -> None:
    from src.aws_burst.image_digest import pinned_image_uri

    cfg = tmp_path / "deploy/aws_burst/config"
    cfg.mkdir(parents=True)
    digest = "sha256:" + "a" * 64
    _write_digest(cfg, "volsurf-burst-2", digest, acct="000000000002")
    uri = pinned_image_uri(tmp_path, "volsurf-burst-2")
    assert "@sha256:" in uri
    assert uri.endswith(digest)
    assert "000000000002" in uri


def test_pinned_image_uri_aborts_when_missing(tmp_path: Path) -> None:
    from src.aws_burst.image_digest import pinned_image_uri

    (tmp_path / "deploy/aws_burst/config").mkdir(parents=True)
    with pytest.raises(ValueError, match="image_digest_missing"):
        pinned_image_uri(tmp_path, "volsurf-burst-1")


def test_deploy_and_teardown_scripts_require_digest_pin() -> None:
    deploy = (ROOT / "deploy/aws_burst/scripts/aws_deploy_batch.sh").read_text()
    teardown = (ROOT / "deploy/aws_burst/scripts/aws_teardown.sh").read_text()
    for script in (deploy, teardown):
        assert "pinned_image_uri" in script
        assert "image_digest_" in script
        assert "@sha256:" in script
        assert "|| true" not in script
        assert "image_uri.txt" not in script


def test_smoke_dry_run_resolves_digest_pin(tmp_path: Path) -> None:
    from scripts.aws_smoke import compose_smoke_plan

    cfg = tmp_path / "deploy/aws_burst/config"
    cfg.mkdir(parents=True)
    digest = "sha256:" + "b" * 64
    _write_digest(cfg, "volsurf-burst-1", digest)
    plan = compose_smoke_plan(profile="volsurf-burst-1", root=tmp_path)
    assert "@sha256:" in plan["image_uri"]
    assert plan["image_uri"].endswith(digest)


# ---------------------------------------------------------------------------
# 32. Job-def image at submit
# ---------------------------------------------------------------------------


def test_preflight_job_definition_image_stale_aborts() -> None:
    from scripts.aws_submit_wave import preflight_job_definition_image

    client = MagicMock()
    client._batch.return_value.describe_job_definitions.return_value = {
        "jobDefinitions": [
            {
                "revision": 9,
                "containerProperties": {
                    "image": "111.dkr.ecr.eu-central-1.amazonaws.com/volsurf-burst@sha256:"
                    + "c" * 64
                },
            }
        ]
    }
    with pytest.raises(ValueError, match="job_definition_image_mismatch"):
        preflight_job_definition_image(
            client, expected_digest="sha256:" + "d" * 64
        )


def test_preflight_job_definition_image_unknown_aborts() -> None:
    from scripts.aws_submit_wave import preflight_job_definition_image

    with pytest.raises(ValueError, match="container_digest_unknown"):
        preflight_job_definition_image(MagicMock(), expected_digest="unknown")


# ---------------------------------------------------------------------------
# 33. Resume digest gate
# ---------------------------------------------------------------------------


def test_resume_digest_mismatch_raises() -> None:
    from deploy.aws_burst.docker import cell_runner

    s3 = MagicMock()
    s3.get_object.return_value = {
        "Body": MagicMock(
            read=lambda: json.dumps({"image_digest": "sha256:aaa"}).encode()
        )
    }
    with patch.object(cell_runner, "_s3_client", return_value=s3):
        with pytest.raises(cell_runner.DigestMismatchError, match="digest_mismatch"):
            cell_runner.assert_resume_digest_compatible(
                "b", "PICK/", "cell_a", "sha256:bbb"
            )


def test_checkpoint_digest_mismatch_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from src.eval.research_alpha_train import _maybe_resume_checkpoint

    class _Agent:
        def load_checkpoint_state(self, *_a, **_k):
            return None

        net = torch.nn.Linear(2, 2)

    ckpt = tmp_path / "fold0_seed0_ep00001.pt"
    torch.save(
        {
            "policy": _Agent().net.state_dict(),
            "seed": 0,
            "fold_id": 0,
            "run_config_hash": "h1",
            "image_digest": "sha256:aaa",
            "episode": 1,
            "optimizer_steps": 1,
        },
        ckpt,
    )
    monkeypatch.setenv("MASCOTRL_CONTAINER_DIGEST", "sha256:bbb")
    agent = _Agent()
    with pytest.raises(RuntimeError, match="digest_mismatch"):
        _maybe_resume_checkpoint(
            agent,
            {
                "_resume_checkpoint": str(ckpt),
                "_run_config_hash": "h1",
            },
        )


# ---------------------------------------------------------------------------
# 34. Sibling archive prefix
# ---------------------------------------------------------------------------


def test_sibling_archive_prefix_and_incomplete_ignores_it() -> None:
    from scripts.aws_submit_wave import _incomplete_cells
    from src.aws_burst.image_digest import sibling_archive_prefix

    ts = "20260825T120000Z"
    assert sibling_archive_prefix("PICK", ts) == f"_archive/PICK_{ts}/"
    client = MagicMock()
    client.account_id.return_value = "111"
    # Sibling archive keys are outside PICK/ listing; nested legacy still ignored.
    client.list_keys.return_value = [
        "PICK/resume/cell_a/cpcv/x.json",
        "PICK/_archive_legacy/cell_a.json",
    ]
    cells = [
        "config/spectrum/cherrypick/cell_a.yaml",
        "config/spectrum/cherrypick/cell_b.yaml",
    ]
    with patch("scripts.aws_submit_wave.artifact_bucket", return_value="arts"):
        incomplete = _incomplete_cells(client, "PICK", cells)
    assert incomplete == cells


# ---------------------------------------------------------------------------
# 35. Armed gate + offline
# ---------------------------------------------------------------------------


def test_armed_unverified_refuses(tmp_path: Path) -> None:
    from src.aws_burst.profiles import armed_profiles

    cfg = tmp_path / "deploy/aws_burst/config"
    cfg.mkdir(parents=True)
    for p in ("volsurf-burst-1", "volsurf-burst-2", "volsurf-burst-3"):
        (cfg / f"budget_armed_{p}.json").write_text(
            '{"verified": false, "action_id": "x", "armed": true}\n',
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="armed_not_verified"):
        armed_profiles(tmp_path)


def test_offline_on_pick_refuses() -> None:
    from scripts import aws_submit_wave as submit_mod

    with pytest.raises(ValueError, match="offline_refused_for_production_wave"):
        submit_mod.submit_wave(ROOT, "PICK", dry_run=True, offline=True)


# ---------------------------------------------------------------------------
# 36. PICK2 gate
# ---------------------------------------------------------------------------


def test_pick2_missing_pick_cell_aborts() -> None:
    from scripts.aws_submit_wave import assert_pick_clean_for_pick2

    client = MagicMock()
    client.profile = "volsurf-burst-1"
    client.account_id.return_value = "111"
    client.list_keys.return_value = []
    with patch("scripts.aws_submit_wave.artifact_bucket", return_value="arts"):
        with pytest.raises(ValueError, match="pick2_gate_missing_pick_cells"):
            assert_pick_clean_for_pick2(
                ROOT, [client], expected_stems=["cell_a", "cell_b"]
            )


def test_pick2_gate_uses_manifest_served_stems_not_full_glob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PICK2 gate must require only cherrypick_final served cells.

    Glob discovery still sees deferred mix/opt and dropped mamba YAMLs under
    config/spectrum/cherrypick/; those must not block PICK2 when the final
    manifest has already dropped them.
    """
    from scripts import aws_submit_wave as submit_mod

    root = tmp_path
    man_dir = root / "config" / "spectrum" / "cherrypick_final"
    man_dir.mkdir(parents=True)
    (man_dir / "manifest.json").write_text(
        json.dumps(
            {
                "cells": ["served_a", "served_b"],
                "dropped_cells": [
                    {"stem": "mix_dropped", "reason": "deferred"},
                    {"stem": "eq_mamba_dropped", "reason": "oom"},
                ],
            }
        ),
        encoding="utf-8",
    )
    # Full glob would invent extra stems; gate must ignore them.
    monkeypatch.setattr(
        submit_mod,
        "discover_wave_cells",
        lambda _r, _w: [
            "config/spectrum/cherrypick/served_a.yaml",
            "config/spectrum/cherrypick/served_b.yaml",
            "config/spectrum/cherrypick/mix_dropped.yaml",
            "config/spectrum/cherrypick/eq_mamba_dropped.yaml",
        ],
    )

    client = MagicMock()
    client.profile = "volsurf-burst-1"
    client.account_id.return_value = "111"
    # Only served finals present.
    client.list_keys.return_value = [
        "PICK/served_a.json",
        "PICK/served_b.json",
    ]
    client.get_json.side_effect = lambda _b, key: {
        "cell_id": Path(key).stem,
        "ok": True,
        "compute_host": "remote",
        "provenance": {"ok": True},
    }

    with patch("scripts.aws_submit_wave.artifact_bucket", return_value="arts"):
        with patch(
            "scripts.validate_remote_cell.validate_remote_cell",
            return_value={"ok": True},
        ):
            # Must not raise for dropped mix/mamba stems absent from S3.
            submit_mod.assert_pick_clean_for_pick2(root, [client])


def test_served_pick_stems_from_manifest_excludes_dropped(tmp_path: Path) -> None:
    from scripts.aws_submit_wave import served_pick_stems_for_pick2_gate

    man_dir = tmp_path / "config" / "spectrum" / "cherrypick_final"
    man_dir.mkdir(parents=True)
    (man_dir / "manifest.json").write_text(
        json.dumps(
            {
                "cells": ["a", "b", "c"],
                "dropped_cells": [{"stem": "b", "reason": "x"}],
            }
        ),
        encoding="utf-8",
    )
    assert served_pick_stems_for_pick2_gate(tmp_path) == ["a", "c"]


# ---------------------------------------------------------------------------
# 37. error.json + stamp failure / logical attempts
# ---------------------------------------------------------------------------


def test_stamp_failure_leaves_no_final_json() -> None:
    from deploy.aws_burst.docker.cell_runner import artifact_missing_provenance

    bad = {
        "compute_host": "remote",
        "provenance_stamp_error": "missing lock",
    }
    assert artifact_missing_provenance(bad) is not None


def test_batch_child_retries_count_as_one_logical_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deploy.aws_burst.docker import cell_runner

    puts: list[dict] = []
    s3 = MagicMock()

    def _get_object(**kwargs):
        if puts:
            return {"Body": MagicMock(read=lambda: puts[-1]["Body"])}
        raise Exception("missing")

    def _put_object(**kwargs):
        puts.append(kwargs)

    s3.get_object.side_effect = _get_object
    s3.put_object.side_effect = _put_object
    monkeypatch.setattr(cell_runner, "_s3_client", lambda: s3)
    monkeypatch.setenv("AWS_BATCH_JOB_ID", "parent123:0")
    env = {
        "MASCOTRL_OUT_URI": "s3://b/PICK/",
        "MASCOTRL_WAVE": "PICK",
        "AWS_BATCH_JOB_ARRAY_INDEX": "0",
        "MASCOTRL_COMPUTE_HOST": "remote",
        "MASCOTRL_CONTAINER_DIGEST": "sha256:abc",
    }
    cell_runner._upload_error(env, "cell_a", RuntimeError("oom"))
    cell_runner._upload_error(env, "cell_a", RuntimeError("oom"))
    cell_runner._upload_error(env, "cell_a", RuntimeError("oom"))
    last = json.loads(puts[-1]["Body"].decode())
    assert last["n_attempts"] == 1

    # New parent job => logical attempt 2; still below max_error_retries=3.
    monkeypatch.setenv("AWS_BATCH_JOB_ID", "parent999:0")
    cell_runner._upload_error(env, "cell_a", RuntimeError("oom"))
    last = json.loads(puts[-1]["Body"].decode())
    assert last["n_attempts"] == 2

    from scripts.aws_submit_wave import _incomplete_cells

    client = MagicMock()
    client.account_id.return_value = "111"
    client.list_keys.return_value = ["PICK/cell_a.error.json"]
    client.get_json.return_value = last
    with patch("scripts.aws_submit_wave.artifact_bucket", return_value="arts"):
        incomplete = _incomplete_cells(
            client, "PICK", ["config/x/cell_a.yaml"], max_error_retries=3
        )
    assert incomplete == ["config/x/cell_a.yaml"]


# ---------------------------------------------------------------------------
# 38. Hash sidecars + fingerprint
# ---------------------------------------------------------------------------


def test_pull_missing_sidecar_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import aws_pull_artifacts as pull_mod

    wave = "PICK_SMOKE"
    cfg = tmp_path / "deploy/aws_burst/config"
    cfg.mkdir(parents=True)
    (cfg / f"wave_{wave}_manifest.json").write_text(
        json.dumps({"expected_cells": ["cell_a"]}) + "\n", encoding="utf-8"
    )
    for p in ("volsurf-burst-1", "volsurf-burst-2", "volsurf-burst-3"):
        (cfg / f"budget_armed_{p}.json").write_text(_armed_payload(p), encoding="utf-8")
        _write_digest(cfg, p, "sha256:" + "e" * 64)

    art_dir = tmp_path / "pull_src"
    art_dir.mkdir()
    art = art_dir / "cell_a.json"
    art.write_text(
        json.dumps(
            {
                "compute_host": "remote",
                "instance_type": "m7i",
                "container_digest": "sha256:" + "e" * 64,
                "requirements_lock_sha256": "f" * 64,
                "universe_fingerprint": "panel",
                "universe_fingerprint_kind": "panel_bundle_sha256",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    # Intentionally no .sha256 sidecar

    class _Client:
        def __init__(self, *_a, **_k):
            pass

        def account_id(self):
            return "1"

        def download_prefix(self, *_a, **_k):
            dest = _k.get("dest") if False else None
            return []

    def _download(self, bucket, prefix, dest, **_kwargs):  # noqa: ANN001
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "cell_a.json").write_bytes(art.read_bytes())
        return [dest / "cell_a.json"]

    monkeypatch.setattr(
        pull_mod,
        "armed_profiles",
        lambda _r: [{"profile": "volsurf-burst-1", "account_id": "1"}],
    )
    monkeypatch.setattr(pull_mod, "BurstClient", type("C", (), {
        "__init__": lambda self, *a, **k: None,
        "account_id": lambda self: "1",
        "download_prefix": _download,
    }))
    out = pull_mod.pull_wave(
        tmp_path, wave, dest=tmp_path / "out", require_complete=False, profiles=["volsurf-burst-1"]
    )
    assert any(r.get("reason") == "missing_sha256_sidecar" for r in out["rejected"])


def test_pull_wrong_fingerprint_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import aws_pull_artifacts as pull_mod

    wave = "PICK_SMOKE"
    cfg = tmp_path / "deploy/aws_burst/config"
    cfg.mkdir(parents=True)
    (cfg / f"wave_{wave}_manifest.json").write_text(
        json.dumps({"expected_cells": ["cell_a"]}) + "\n", encoding="utf-8"
    )
    for p in ("volsurf-burst-1", "volsurf-burst-2", "volsurf-burst-3"):
        (cfg / f"budget_armed_{p}.json").write_text(_armed_payload(p), encoding="utf-8")
        _write_digest(cfg, p, "sha256:" + "e" * 64)

    payload = {
        "compute_host": "remote",
        "instance_type": "m7i",
        "container_digest": "sha256:" + "e" * 64,
        "requirements_lock_sha256": "f" * 64,
        "universe_fingerprint": "wrong-fp",
        "universe_fingerprint_kind": "panel_bundle_sha256",
    }
    body = (json.dumps(payload) + "\n").encode()
    digest = hashlib.sha256(body).hexdigest()

    def _download(self, bucket, prefix, dest, **_kwargs):  # noqa: ANN001
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "cell_a.json").write_bytes(body)
        (dest / "cell_a.json.sha256").write_text(digest + "\n", encoding="utf-8")
        return [dest / "cell_a.json"]

    monkeypatch.setattr(
        pull_mod,
        "armed_profiles",
        lambda _r: [{"profile": "volsurf-burst-1", "account_id": "1"}],
    )
    monkeypatch.setattr(
        pull_mod,
        "BurstClient",
        type(
            "C",
            (),
            {
                "__init__": lambda self, *a, **k: None,
                "account_id": lambda self: "1",
                "download_prefix": _download,
            },
        ),
    )
    out = pull_mod.pull_wave(
        tmp_path,
        wave,
        dest=tmp_path / "out",
        require_complete=False,
        profiles=["volsurf-burst-1"],
        expected_universe_fingerprint="expected-fp",
        expected_container_digest="sha256:" + "e" * 64,
    )
    assert out["n_accepted"] == 0
    assert any("universe_fingerprint" in str(r.get("reason")) for r in out["rejected"])


# ---------------------------------------------------------------------------
# 39. Watcher exact stem set
# ---------------------------------------------------------------------------


def test_watcher_wrong_stem_extra_does_not_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import aws_burst_s3_watch as watch_mod

    class _FakeS3:
        def get_paginator(self, _name):
            return self

        def paginate(self, **kwargs):
            return [
                {
                    "Contents": [
                        {"Key": "PICK/cell_a.json", "Size": 10},
                        {"Key": "PICK/cell_extra.json", "Size": 10},
                    ]
                }
            ]

        def get_object(self, **kwargs):
            art = {
                "compute_host": "remote",
                "instance_type": "m7i",
                "container_digest": "sha256:abc",
                "requirements_lock_sha256": "x" * 64,
                "universe_fingerprint": "fp",
                "universe_fingerprint_kind": "panel_bundle_sha256",
            }
            return {"Body": MagicMock(read=lambda: json.dumps(art).encode())}

    class _Sess:
        def __init__(self, *a, **k):
            pass

        def client(self, *_a, **_k):
            return _FakeS3()

    monkeypatch.setattr(watch_mod.boto3, "Session", _Sess)
    snap = watch_mod.count_wave_root_finals(
        "PICK",
        profiles=["volsurf-burst-1"],
        expected_stems=["cell_a"],
        validate=True,
    )
    assert snap["complete"] is False
    assert snap["n_extras"] == 1


# ---------------------------------------------------------------------------
# 40. Shell soft-fail removed
# ---------------------------------------------------------------------------


def test_shell_scripts_no_soft_true_masking() -> None:
    for rel in (
        "deploy/aws_burst/scripts/aws_teardown.sh",
        "deploy/aws_burst/scripts/aws_request_quotas.sh",
        "deploy/aws_burst/scripts/aws_poll_quotas.sh",
    ):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "|| true" not in text
        assert "set -euo pipefail" in text


def test_count_active_pick2_jobs_filters_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import aws_pick2_resume_loop as loop

    class _C:
        def __init__(self, profile: str, _region: str) -> None:
            self.profile = profile

        def list_jobs(self, _queue: str, status: str) -> list[dict]:
            if status != "RUNNING":
                return []
            if self.profile == "volsurf-burst-1":
                return [
                    {"jobName": "mascotrl-PICK2-abcd1234-volsurf-burst-1"},
                    {"jobName": "mascotrl-PICK-ffff-volsurf-burst-1"},
                ]
            return [{"jobName": "other"}]

    monkeypatch.setattr(loop, "BurstClient", _C)
    monkeypatch.setattr(
        loop,
        "armed_profiles",
        lambda _r: [{"profile": "volsurf-burst-1"}, {"profile": "volsurf-burst-2"}],
    )
    assert loop.count_active_pick2_jobs(ROOT) == 1
