"""AWS panel-bundle tests: mirror-first ship list, manifest sha, submit gate."""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.plumbing


def test_required_panel_families_lists_eleven_plus_gics() -> None:
    from mascotrl.data.feature_panels import (
        FAMILY_LOADERS,
        FAMILY_RAW_FALLBACKS,
        GICS_RAW_PATHS,
        required_panel_families,
        required_panel_mirror_names,
    )

    families = required_panel_families()
    mirrors = required_panel_mirror_names()
    assert len(families) == 11
    assert set(families) == set(FAMILY_LOADERS)
    assert mirrors == [f"feat_{f}.parquet" for f in families]
    assert "macro/lseg_gics.parquet" in GICS_RAW_PATHS
    assert "macro/lseg_ric_map.parquet" in GICS_RAW_PATHS
    for fam in families:
        assert fam in FAMILY_RAW_FALLBACKS
        assert FAMILY_RAW_FALLBACKS[fam]


def test_collect_panel_bundle_paths_mirror_first(tmp_path: Path) -> None:
    from mascotrl.data.feature_panels import collect_panel_bundle_paths

    lake = tmp_path / "lake"
    (lake / "_panels").mkdir(parents=True)
    (lake / "macro").mkdir(parents=True)
    (lake / "macro" / "p3").mkdir(parents=True)
    (lake / "factors").mkdir(parents=True)

    # Only ohlc mirror present; microstructure needs raw fallback.
    (lake / "_panels" / "feat_ohlc.parquet").write_bytes(b"ohlc")
    (lake / "macro" / "lseg_eq_ohlc_unadj.parquet").write_bytes(b"raw")
    (lake / "macro" / "lseg_eq_size.parquet").write_bytes(b"size")
    (lake / "macro" / "lseg_gics.parquet").write_bytes(b"gics")
    (lake / "macro" / "lseg_ric_map.parquet").write_bytes(b"ric")
    (lake / "macro" / "ff_factors.parquet").write_bytes(b"ff")

    entries = collect_panel_bundle_paths(lake, require_complete=False)
    rels = {e["arcname"] for e in entries}
    assert "_panels/feat_ohlc.parquet" in rels
    assert "macro/lseg_eq_ohlc_unadj.parquet" in rels  # microstructure fallback
    assert "macro/lseg_gics.parquet" in rels
    assert "macro/lseg_ric_map.parquet" in rels
    assert "macro/ff_factors.parquet" in rels
    # Mirror preferred: no raw ohlc-only paths required beyond shared unadj.
    ohlc_entry = next(e for e in entries if e["arcname"] == "_panels/feat_ohlc.parquet")
    assert ohlc_entry["source"] == "mirror"


def test_collect_panel_bundle_paths_require_complete_fails(tmp_path: Path) -> None:
    from mascotrl.data.feature_panels import collect_panel_bundle_paths

    lake = tmp_path / "lake"
    lake.mkdir()
    with pytest.raises(ValueError, match="panel_bundle_missing_families"):
        collect_panel_bundle_paths(lake, require_complete=True)


def test_build_panel_bundle_writes_manifest(tmp_path: Path) -> None:
    from mascotrl.aws_burst.panel_bundle import build_panel_bundle
    from mascotrl.data.feature_panels import required_panel_families

    lake = tmp_path / "lake"
    arctic = tmp_path / "arctic"
    arctic.mkdir()
    (arctic / "marker").write_text("ok")
    (lake / "_panels").mkdir(parents=True)
    (lake / "macro").mkdir(parents=True)
    (lake / "macro" / "p3").mkdir()
    (lake / "factors").mkdir()
    (lake / "surface_signals").mkdir()
    for fam in required_panel_families():
        (lake / "_panels" / f"feat_{fam}.parquet").write_bytes(fam.encode())
    (lake / "macro" / "lseg_gics.parquet").write_bytes(b"g")
    (lake / "macro" / "lseg_ric_map.parquet").write_bytes(b"r")
    (lake / "macro" / "ff_factors.parquet").write_bytes(b"ff")
    # Substrate parity files (sp500_sec + geometry_lite cache).
    (lake / "macro" / "sp500_sec.parquet").write_bytes(b"sp500")
    (lake / "surface_signals" / "geometry_lite.parquet").write_bytes(b"surf")

    out = tmp_path / "out"
    meta = build_panel_bundle(
        lake=lake,
        arctic=arctic,
        out_dir=out,
        require_complete=True,
    )
    assert (out / "panel_bundle.tar").is_file()
    assert (out / "panel_bundle.sha256").is_file()
    manifest = json.loads((out / "panel_manifest.json").read_text())
    assert manifest["sha256"] == meta["sha256"]
    assert manifest["files"]
    assert all("sha256" in f for f in manifest["files"])
    with tarfile.open(out / "panel_bundle.tar", "r") as tar:
        names = tar.getnames()
    assert "panel_manifest.json" in names
    assert any(n.startswith("_panels/feat_") for n in names)
    assert "macro/sp500_sec.parquet" in names
    assert "surface_signals/geometry_lite.parquet" in names
    assert "volsurf_arcticdb/marker" in names or any(
        "volsurf_arcticdb" in n for n in names
    )


def test_collect_panel_bundle_paths_require_substrate(tmp_path: Path) -> None:
    """require_complete fails closed when sp500_sec / geometry_lite absent."""
    from mascotrl.data.feature_panels import (
        collect_panel_bundle_paths,
        required_panel_families,
    )

    lake = tmp_path / "lake"
    (lake / "_panels").mkdir(parents=True)
    (lake / "macro").mkdir(parents=True)
    for fam in required_panel_families():
        (lake / "_panels" / f"feat_{fam}.parquet").write_bytes(fam.encode())
    (lake / "macro" / "lseg_gics.parquet").write_bytes(b"g")
    (lake / "macro" / "lseg_ric_map.parquet").write_bytes(b"r")
    with pytest.raises(ValueError, match="panel_bundle_missing_substrate"):
        collect_panel_bundle_paths(lake, require_complete=True)


def test_assert_wave_panel_families_refuse_missing(tmp_path: Path) -> None:
    from scripts.aws_submit_wave import assert_wave_panel_families_present

    # Cell requiring feature cube
    cell = tmp_path / "cell.yaml"
    cell.write_text("use_equity_feature_cube: true\nalgo: ppo\n")
    manifest = {
        "families_present": ["ohlc"],  # incomplete
        "files": [{"arcname": "_panels/feat_ohlc.parquet"}],
    }
    with pytest.raises(ValueError, match="panel_bundle_missing_required_families"):
        assert_wave_panel_families_present(
            [str(cell)],
            manifest,
            root=tmp_path,
        )


def test_cell_runner_asserts_panel_manifest(tmp_path: Path) -> None:
    from deploy.aws_burst.docker.cell_runner import assert_panel_manifest

    extract = tmp_path / "panel"
    extract.mkdir()
    with pytest.raises(RuntimeError, match="panel_manifest_missing"):
        assert_panel_manifest(extract)

    (extract / "panel_manifest.json").write_text("{}")
    with pytest.raises(RuntimeError, match="panel_manifest_empty_files"):
        assert_panel_manifest(extract)

    man = {
        "files": [
            {"arcname": "_panels/feat_ohlc.parquet", "sha256": "abc", "bytes": 1},
        ]
    }
    (extract / "panel_manifest.json").write_text(json.dumps(man))
    (extract / "_panels").mkdir()
    (extract / "_panels" / "feat_ohlc.parquet").write_bytes(b"x")
    # sha mismatch
    with pytest.raises(RuntimeError, match="panel_manifest_sha_mismatch"):
        assert_panel_manifest(extract)

    digest = hashlib.sha256(b"x").hexdigest()
    man["files"][0]["sha256"] = digest
    (extract / "panel_manifest.json").write_text(json.dumps(man))
    assert_panel_manifest(extract)  # no raise
