#!/usr/bin/env python3
"""Aggregate per-cell policy_behavior JSON into a panel directory for B-figures."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SOURCES = (
    ROOT / "logs/artifacts/spectrum/fullgrid",
    ROOT / "logs/artifacts/spectrum/cherrypick",
    ROOT / "logs/artifacts/eq_alloc",
)


def discover_behavior_files(sources: list[Path]) -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()
    for src in sources:
        if not src.is_dir():
            if src.is_file() and src.name.endswith("_policy_behavior.json"):
                key = str(src.resolve())
                if key not in seen:
                    seen.add(key)
                    found.append(src)
            continue
        for path in sorted(src.rglob("*_policy_behavior.json")):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            found.append(path)
        single = src / "policy_behavior.json"
        if single.is_file():
            key = str(single.resolve())
            if key not in seen:
                seen.add(key)
                found.append(single)
    return found


def discover_interpretability_files(sources: list[Path]) -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()
    for src in sources:
        if not src.is_dir():
            continue
        for path in sorted(src.rglob("*_interpretability.json")):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            found.append(path)
    return found


def build_panel(
    *,
    sources: list[Path],
    out_dir: Path,
    link: bool = True,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = discover_behavior_files(sources)
    written: list[str] = []
    for src in files:
        dest = out_dir / src.name
        if link:
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            dest.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dest)
        written.append(str(dest))
    interpret_files = discover_interpretability_files(sources)
    manifest = {
        "schema": "mascotrl.behaviour_panel.v1",
        "n_cells": len(written),
        "sources": [str(s) for s in sources],
        "files": written,
        "interpretability_files": [str(p) for p in interpret_files],
    }
    (out_dir / "panel_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sources",
        type=Path,
        nargs="*",
        default=list(DEFAULT_SOURCES),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "logs/artifacts/policy_behavior_panel",
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="copy files instead of symlinking",
    )
    args = p.parse_args(argv)
    manifest = build_panel(
        sources=list(args.sources),
        out_dir=args.out,
        link=not bool(args.copy),
    )
    print(f"wrote panel n_cells={manifest['n_cells']} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
