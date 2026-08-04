"""Part D.4 companion: remote cell validation script surface."""
from __future__ import annotations

from scripts.validate_remote_cell import REQUIRED_REMOTE_FIELDS, validate_remote_cell


def test_required_remote_fields_documented() -> None:
    assert "compute_host" in REQUIRED_REMOTE_FIELDS
    assert "instance_type" in REQUIRED_REMOTE_FIELDS
    assert "container_digest" in REQUIRED_REMOTE_FIELDS
    assert "requirements_lock_sha256" in REQUIRED_REMOTE_FIELDS
    # crucible_fingerprint is optional when universe_fingerprint+kind is used (WP-R2)
    assert "crucible_fingerprint" not in REQUIRED_REMOTE_FIELDS


def test_valid_remote_artifact_retains_provenance() -> None:
    art = {
        "compute_host": "remote",
        "instance_type": "m5.2xlarge",
        "container_digest": "sha256:fff",
        "requirements_lock_sha256": "abc123",
        "crucible_fingerprint": "crucible-xyz",
        "extra_note": "keep-me",
    }
    out = validate_remote_cell(art, expected_fingerprint="crucible-xyz")
    assert out["ok"] is True
    assert out["artifact"]["extra_note"] == "keep-me"
    assert out["artifact"]["instance_type"] == "m5.2xlarge"
