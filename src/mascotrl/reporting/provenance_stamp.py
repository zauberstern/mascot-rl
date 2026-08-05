"""Remote provenance stamp (WP-R1). Refuse partial stamps."""
from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

REQUIRED_STAMP_FIELDS = (
    "compute_host",
    "instance_type",
    "container_digest",
    "requirements_lock_sha256",
)


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_imds_instance_type(*, timeout_s: float = 1.0) -> str | None:
    """IMDSv2 instance type; returns None when not on EC2."""
    try:
        token_req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        with urllib.request.urlopen(token_req, timeout=timeout_s) as resp:
            token = resp.read().decode("utf-8")
        type_req = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/instance-type",
            headers={"X-aws-ec2-metadata-token": token},
        )
        with urllib.request.urlopen(type_req, timeout=timeout_s) as resp:
            return resp.read().decode("utf-8").strip() or None
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def build_provenance_stamp(
    *,
    requirements_lock_path: Path | str | None = None,
    container_digest: str | None = None,
    instance_type: str | None = None,
    compute_host: str = "remote",
    universe_fingerprint: str | None = None,
    universe_fingerprint_kind: str | None = None,
    crucible_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Assemble a complete stamp or raise ``ValueError`` (refuse partial)."""
    lock_path = Path(
        requirements_lock_path
        or os.environ.get(
            "MASCOTRL_REQUIREMENTS_LOCK",
            "deploy/training_remote/requirements.lock",
        )
    )
    if not lock_path.is_file():
        raise ValueError(f"provenance_incomplete: missing lock file {lock_path}")
    lock_sha = sha256_file(lock_path)
    digest = str(container_digest or os.environ.get("MASCOTRL_CONTAINER_DIGEST") or "").strip()
    itype = str(
        instance_type
        or os.environ.get("MASCOTRL_INSTANCE_TYPE")
        or fetch_imds_instance_type()
        or ""
    ).strip()
    stamp: dict[str, Any] = {
        "compute_host": str(compute_host),
        "instance_type": itype,
        "container_digest": digest,
        "requirements_lock_sha256": lock_sha,
    }
    missing = [k for k in REQUIRED_STAMP_FIELDS if not stamp.get(k)]
    if missing:
        raise ValueError(f"provenance_incomplete: missing fields {missing}")
    if universe_fingerprint:
        stamp["universe_fingerprint"] = str(universe_fingerprint)
        stamp["universe_fingerprint_kind"] = str(
            universe_fingerprint_kind or "dyn_hrp"
        )
    if crucible_fingerprint:
        stamp["crucible_fingerprint"] = str(crucible_fingerprint)
    return stamp


def write_provenance_stamp(path: Path | str, stamp: Mapping[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(stamp), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p
