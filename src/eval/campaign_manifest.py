"""Incremental CPCV campaign manifest: skip completed (fold, seed, arm) cells."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


MANIFEST_NAME = "campaign_manifest.json"

# Batch disk writes during CPCV (mark_cell_complete stays in-memory every fold).
MANIFEST_FLUSH_EVERY = int(os.environ.get("MASCOTRL_MANIFEST_FLUSH_EVERY", "8"))


class ManifestFlushWriter:
    """Defer ``save_manifest`` until every N folds or explicit flush."""

    __slots__ = ("_out_dir", "_flush_every", "_pending")

    def __init__(
        self,
        out_dir: str | Path,
        *,
        flush_every: int | None = None,
    ) -> None:
        self._out_dir = Path(out_dir)
        self._flush_every = max(1, int(flush_every or MANIFEST_FLUSH_EVERY))
        self._pending = 0

    def mark_and_maybe_flush(
        self,
        manifest: dict[str, Any],
        *,
        fold_id: int,
        last_fold_id: int,
        seed: int,
        arm: str,
        pnl: Mapping[str, float] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        mark_cell_complete(
            manifest,
            fold_id,
            seed,
            arm,
            pnl=pnl,
            extra=extra,
        )
        self._pending += 1
        if self._pending >= self._flush_every or int(fold_id) == int(last_fold_id):
            save_manifest(self._out_dir, manifest)
            self._pending = 0

    def flush(self, manifest: dict[str, Any]) -> None:
        if self._pending > 0:
            save_manifest(self._out_dir, manifest)
            self._pending = 0


def cell_key(fold_id: int, seed: int, arm: str) -> str:
    """Stable key for a CPCV campaign cell."""
    return f"{int(fold_id)}|{int(seed)}|{str(arm)}"


def manifest_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / MANIFEST_NAME


def load_manifest(out_dir: str | Path) -> dict[str, Any]:
    path = manifest_path(out_dir)
    if not path.exists():
        return {"version": 1, "completed": {}}
    try:
        blob = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "completed": {}}
    if not isinstance(blob, dict):
        return {"version": 1, "completed": {}}
    completed = blob.get("completed")
    if not isinstance(completed, dict):
        blob["completed"] = {}
    blob.setdefault("version", 1)
    return blob


def is_cell_complete(
    manifest: Mapping[str, Any],
    fold_id: int,
    seed: int,
    arm: str,
) -> bool:
    completed = manifest.get("completed") or {}
    entry = completed.get(cell_key(fold_id, seed, arm))
    if not isinstance(entry, Mapping):
        return False
    return bool(entry.get("ok") or entry.get("completed")) and bool(
        isinstance(entry.get("pnl"), Mapping) and len(entry.get("pnl") or {}) > 0
    )


def get_completed_pnl(
    manifest: Mapping[str, Any],
    fold_id: int,
    seed: int,
    arm: str,
) -> dict[str, float] | None:
    completed = manifest.get("completed") or {}
    entry = completed.get(cell_key(fold_id, seed, arm))
    if not isinstance(entry, Mapping):
        return None
    pnl = entry.get("pnl")
    if not isinstance(pnl, Mapping):
        return None
    return {str(k): float(v) for k, v in pnl.items()}


def get_completed_extra(
    manifest: Mapping[str, Any],
    fold_id: int,
    seed: int,
    arm: str,
) -> dict[str, Any] | None:
    """Return the optional ``extra`` blob cached with a completed fold cell."""
    completed = manifest.get("completed") or {}
    entry = completed.get(cell_key(fold_id, seed, arm))
    if not isinstance(entry, Mapping):
        return None
    extra = entry.get("extra")
    if not isinstance(extra, Mapping):
        return None
    return dict(extra)


def mark_cell_complete(
    manifest: dict[str, Any],
    fold_id: int,
    seed: int,
    arm: str,
    *,
    pnl: Mapping[str, float] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    completed = dict(manifest.get("completed") or {})
    entry: dict[str, Any] = {
        "ok": True,
        "completed": True,
        "fold_id": int(fold_id),
        "seed": int(seed),
        "arm": str(arm),
    }
    if pnl is not None:
        entry["pnl"] = {str(k): float(v) for k, v in pnl.items()}
    if extra:
        entry["extra"] = dict(extra)
    completed[cell_key(fold_id, seed, arm)] = entry
    manifest["completed"] = completed
    return manifest


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write JSON via temp file + os.replace (atomic on same filesystem)."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        dir=str(dest.parent),
        prefix=f".{dest.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, dest)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def save_manifest(out_dir: str | Path, manifest: Mapping[str, Any]) -> Path:
    path = manifest_path(out_dir)
    atomic_write_json(path, manifest)
    return path


def purge_orphan_fold_cells(
    manifest: dict[str, Any],
    *,
    out_dir: str | Path,
    arm: str,
    seeds: Sequence[int],
) -> list[str]:
    """Drop fold-level resume cells whose seed never produced an artifact.

    Overnight crashes after fold caching but before ``cpcv_seed_N.json`` leave
    fold completions that later resume-skip every fold with
    ``optimizer_steps=0``, tripping the train-budget floor. Removing those
    orphans forces a real retrain for seeds that are not yet sealed.
    """
    completed = dict(manifest.get("completed") or {})
    removed: list[str] = []
    arm_s = str(arm)
    out = Path(out_dir)
    for seed in seeds:
        seed_i = int(seed)
        seed_art = out / f"cpcv_seed_{seed_i}.json"
        seed_cell = cell_key(-1, seed_i, arm_s)
        seed_ok = seed_art.is_file() and is_cell_complete(manifest, -1, seed_i, arm_s)
        if seed_ok:
            continue
        for key in list(completed):
            parts = str(key).split("|")
            if len(parts) != 3:
                continue
            fold_s, seed_s, arm_part = parts
            try:
                fold_i = int(fold_s)
                seed_part = int(seed_s)
            except ValueError:
                continue
            if fold_i < 0:
                continue
            if seed_part == seed_i and arm_part == arm_s:
                completed.pop(key, None)
                removed.append(key)
        if seed_cell in completed and not seed_art.is_file():
            completed.pop(seed_cell, None)
            removed.append(seed_cell)
    if removed:
        manifest["completed"] = completed
    return removed
