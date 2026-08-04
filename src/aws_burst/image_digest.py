"""Per-profile ECR digest pins for Batch ImageUri (fail-closed)."""
from __future__ import annotations

import json
from pathlib import Path


DIGEST_DIR = Path("deploy/aws_burst/config")


def digest_file_path(root: Path, profile: str) -> Path:
    return Path(root) / DIGEST_DIR / f"image_digest_{profile}.json"


def load_digest_record(root: Path, profile: str) -> dict[str, str]:
    path = digest_file_path(root, profile)
    if not path.is_file():
        raise ValueError(f"image_digest_missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"image_digest_corrupt: {path}: {exc}") from exc
    digest = str(data.get("digest") or "").strip()
    image = str(data.get("image") or "").strip()
    if not digest.startswith("sha256:"):
        raise ValueError(f"image_digest_invalid: profile={profile} digest={digest!r}")
    if not image:
        raise ValueError(f"image_digest_missing_image: profile={profile}")
    return {"profile": profile, "digest": digest, "image": image, "path": str(path)}


def pinned_image_uri(root: Path, profile: str) -> str:
    """Return ``repo@sha256:...`` for CFN ImageUri / job-def checks."""
    rec = load_digest_record(root, profile)
    digest = rec["digest"]
    image = rec["image"]
    if "@sha256:" in image:
        base = image.split("@", 1)[0]
        return f"{base}@{digest}"
    # ``acct.../volsurf-burst:latest`` -> strip tag
    if ":" in image.rsplit("/", 1)[-1]:
        base = image.rsplit(":", 1)[0]
    else:
        base = image
    uri = f"{base}@{digest}"
    if "@sha256:" not in uri:
        raise ValueError(f"image_uri_not_digest_pinned: {uri!r}")
    return uri


def sibling_archive_prefix(wave: str, ts: str) -> str:
    """Archive target sibling of ``{wave}/`` (not nested under the wave)."""
    return f"_archive/{wave}_{ts}/"
