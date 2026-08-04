"""Regression: stop-verify must prefer the project venv for pytest."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / ".cursor" / "hooks" / "resolve_pytest_cmd.py"


def _load_resolve():
    spec = importlib.util.spec_from_file_location("resolve_pytest_cmd", HOOK)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_resolve_pytest_cmd_prefers_venv_python():
    mod = _load_resolve()
    venv_python = ROOT / ".venv" / "bin" / "python"
    if not (venv_python.is_file() and os.access(venv_python, os.X_OK)):
        # CI / fresh clones may not have a local venv; skip only then.
        import pytest

        pytest.skip("project .venv not present")
    cmd = mod.resolve_pytest_cmd(str(ROOT), extra=["-q"])
    assert cmd[0] == str(venv_python)
    assert cmd[1:3] == ["-m", "pytest"]


def test_core_deps_importable_under_active_interpreter():
    """Fail loudly if pytest is not the project env (torch / arcticdb)."""
    import importlib

    missing = []
    for name in ("torch", "arcticdb", "duckdb"):
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    assert not missing, (
        "Missing packages under this interpreter: "
        + ", ".join(missing)
        + ". Run: .venv/bin/python -m pytest -q"
    )
