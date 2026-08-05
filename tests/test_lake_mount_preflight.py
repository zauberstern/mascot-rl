"""Lake mount preflight must refuse unmounted USB / missing lake root."""
from __future__ import annotations

from pathlib import Path

import pytest


def test_assert_lake_mounted_raises_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mascotrl.data import paths as P

    missing = tmp_path / "no_lake"
    monkeypatch.setattr(P, "LAKE_ROOT", missing)
    monkeypatch.setattr(P, "CANONICAL_LAKE", tmp_path / "other")
    with pytest.raises(SystemExit, match="lake root missing"):
        P.assert_lake_mounted(missing)


def test_ensure_lake_dirs_does_not_create_canonical_on_unmounted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mascotrl.data import paths as P

    fake_mount = tmp_path / "not_mounted"
    fake_lake = fake_mount / "volsurf_data_lake"
    monkeypatch.setattr(P, "MOUNT_ROOT", fake_mount)
    monkeypatch.setattr(P, "CANONICAL_LAKE", fake_lake)
    monkeypatch.setattr(P, "LAKE_ROOT", fake_lake)
    with pytest.raises(SystemExit, match="lake mount missing"):
        P.ensure_lake_dirs()
    assert not fake_lake.exists()
