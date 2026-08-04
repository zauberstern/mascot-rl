"""C-2: aws_pull_artifacts CLI universe-fingerprint validation."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.plumbing

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_pull(*args: str) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "aws_pull_artifacts.py"),
        *args,
        "--allow-incomplete",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_eq_wave_without_fingerprint_exits_2(tmp_path: Path, monkeypatch) -> None:
    from scripts import aws_pull_artifacts as pull_mod

    monkeypatch.setattr(pull_mod, "ROOT", tmp_path)
    code = pull_mod.main(["--wave", "PICK", "--allow-incomplete"])
    assert code == 2


def test_eq_wave_with_explicit_fingerprint_does_not_fail_cli_validation(
    tmp_path: Path, monkeypatch,
) -> None:
    from scripts import aws_pull_artifacts as pull_mod

    monkeypatch.setattr(pull_mod, "ROOT", tmp_path)
    monkeypatch.setattr(
        pull_mod,
        "pull_wave",
        lambda *a, **k: {"n_accepted": 0, "rejected": [], "complete": True},
    )
    code = pull_mod.main(
        [
            "--wave",
            "PICK_SMOKE",
            "--expected-universe-fingerprint",
            "deadbeef",
            "--allow-incomplete",
        ]
    )
    assert code == 0


def test_allow_missing_universe_fingerprint_skips_requirement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts import aws_pull_artifacts as pull_mod

    monkeypatch.setattr(pull_mod, "ROOT", tmp_path)
    monkeypatch.setattr(
        pull_mod,
        "pull_wave",
        lambda *a, **k: {"n_accepted": 0, "rejected": [], "complete": True},
    )
    code = pull_mod.main(
        ["--wave", "K200", "--allow-missing-universe-fingerprint", "--allow-incomplete"]
    )
    assert code == 0


def test_resolve_expected_universe_fingerprint_uses_panel_bundle(tmp_path: Path) -> None:
    from scripts.aws_pull_artifacts import resolve_expected_universe_fingerprint

    bundle_dir = tmp_path / "logs/aws_burst_panel_bundle"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "panel_bundle.sha256").write_text("abc123\n", encoding="utf-8")
    fp = resolve_expected_universe_fingerprint(tmp_path, "PICK")
    assert fp == "abc123"


def test_pull_validation_refuses_universe_fingerprint_mismatch() -> None:
    """Regression guard: validate_remote_cell mismatch is surfaced on pull."""
    from scripts.validate_remote_cell import validate_remote_cell

    art = {
        "compute_host": "remote",
        "instance_type": "m7i",
        "container_digest": "sha256:" + "a" * 64,
        "requirements_lock_sha256": "b" * 64,
        "universe_fingerprint": "wrong",
        "universe_fingerprint_kind": "panel_bundle_sha256",
    }
    res = validate_remote_cell(art, expected_universe_fingerprint="expected")
    assert res["ok"] is False
    assert any("universe_fingerprint" in e for e in res["errors"])
