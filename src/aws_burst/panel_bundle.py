"""AWS Burst panel bundle: feature-cube lake files + Arctic + manifest."""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

from src.data.feature_panels import collect_panel_bundle_paths, required_panel_families

ROOT = Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_panel_bundle(
    *,
    lake: Path | str,
    arctic: Path | str | None,
    out_dir: Path | str,
    require_complete: bool = True,
    max_bytes: int = 2 * (1 << 30),
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Write panel_bundle.tar + sha256 + panel_manifest.json under out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(repo_root) if repo_root is not None else ROOT
    lake_p = Path(lake)
    entries = collect_panel_bundle_paths(lake_p, require_complete=require_complete)

    file_rows: list[dict[str, Any]] = []
    for entry in entries:
        digest = _sha256_file(entry["abs_path"])
        file_rows.append(
            {
                "arcname": entry["arcname"],
                "sha256": digest,
                "bytes": int(entry["abs_path"].stat().st_size),
                "source": entry["source"],
                "family": entry["family"],
            }
        )

    families_present = sorted(
        {str(r["family"]) for r in file_rows if r.get("family")}
    )
    manifest: dict[str, Any] = {
        "version": 1,
        "required_families": required_panel_families(),
        "families_present": families_present,
        "files": file_rows,
    }
    manifest_path = out / "panel_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    tar_path = out / "panel_bundle.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(manifest_path, arcname="panel_manifest.json")
        arctic_p = Path(arctic) if arctic else None
        if arctic_p is not None and arctic_p.is_dir():
            tar.add(arctic_p, arcname="volsurf_arcticdb")
        for entry in entries:
            tar.add(entry["abs_path"], arcname=entry["arcname"])
        lock = root / "deploy" / "training_remote" / "requirements.lock"
        if lock.is_file():
            tar.add(lock, arcname="requirements.lock")

    digest = _sha256_file(tar_path)
    (out / "panel_bundle.sha256").write_text(digest + "\n")
    manifest["sha256"] = digest
    manifest["tar_bytes"] = int(tar_path.stat().st_size)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if tar_path.stat().st_size > max_bytes:
        raise SystemExit(
            f"bundle exceeds {max_bytes} bytes ({tar_path.stat().st_size}); refuse"
        )
    meta = {
        "sha256": digest,
        "tar": str(tar_path),
        "bytes": int(tar_path.stat().st_size),
        "manifest": str(manifest_path),
        "families_present": families_present,
    }
    (out / "panel_bundle_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
