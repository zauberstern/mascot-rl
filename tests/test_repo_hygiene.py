"""Repo hygiene guards after the less-is-more overhaul."""
from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

from src.aws_burst.profiles import BURST_PROFILES, armed_profiles
from src.aws_burst.waves import WAVES
from src.data.paths import MASCOTRL_ROOT
from src.eval.cpcv import CPCVConfig

SCAN_ROOTS = ("src", "scripts", "tests")
SKIP_DIRS = {".venv", ".git", "graveyard", "authority", "logs", "node_modules"}

# Renamed-away module path fragments (import statements only).
BANNED_MODULE_FRAGMENTS = (
    "rank1_claim_stamps",
    "dh_option_allocator_stamps",
    "capital_hygiene",
    "academic_style",
    "behaviour_metrics",
    "behaviour_explain",
    "figures.style",
    "render_all",
    "aws_burst.provenance",
)

UK_BEHAVIOUR_MODULE = re.compile(r"(^|/)behaviour[^/]*\.py$")


def _iter_py_under(*roots: str) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        base = MASCOTRL_ROOT / root
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            out.append(path)
    return out


def _import_module_names(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_no_banned_module_imports() -> None:
    hits: list[str] = []
    for path in _iter_py_under(*SCAN_ROOTS):
        rel = path.relative_to(MASCOTRL_ROOT).as_posix()
        if rel == "tests/test_repo_hygiene.py":
            continue
        for mod in _import_module_names(path):
            for frag in BANNED_MODULE_FRAGMENTS:
                if frag in mod:
                    hits.append(f"{rel}: imports {mod!r}")
                    break
    assert hits == [], "banned module imports:\n" + "\n".join(hits)


def test_no_uk_behaviour_module_filenames() -> None:
    hits: list[str] = []
    for path in _iter_py_under(*SCAN_ROOTS):
        rel = path.relative_to(MASCOTRL_ROOT).as_posix()
        if UK_BEHAVIOUR_MODULE.search(rel):
            hits.append(rel)
    assert hits == [], "UK behaviour module filenames:\n" + "\n".join(hits)


def test_root_community_files_present() -> None:
    # README.md ships in Phase 9; LICENSE is required in the public extract.
    assert (MASCOTRL_ROOT / "LICENSE").is_file(), "missing root LICENSE"


def test_graveyard_not_tracked_by_git() -> None:
    git_dir = MASCOTRL_ROOT / ".git"
    if not git_dir.is_dir():
        return
    proc = subprocess.run(
        ["git", "ls-files", "graveyard/"],
        cwd=MASCOTRL_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    tracked = [line for line in proc.stdout.splitlines() if line.strip()]
    assert tracked == [], f"graveyard/ must not be tracked:\n" + "\n".join(tracked)


def test_cpcv_config_defaults_purge_embargo_21() -> None:
    cfg = CPCVConfig()
    assert cfg.purge_days == 21
    assert cfg.embargo_days == 21


def test_b200_removed_or_distinct_from_e200() -> None:
    assert "B200" not in WAVES
    assert WAVES["E200"].glob == "config/spectrum/fullgrid/*_K200_*.yaml"


def test_armed_profiles_requires_three_files(tmp_path: Path) -> None:
    cfg = tmp_path / "deploy" / "aws_burst" / "config"
    cfg.mkdir(parents=True)
    for p in BURST_PROFILES:
        (cfg / f"budget_armed_{p['profile']}.json").write_text(
            json.dumps(
                {
                    "verified": True,
                    "action_id": f"act-{p['profile']}",
                    "armed": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    armed = armed_profiles(tmp_path)
    assert len(armed) == len(BURST_PROFILES)
