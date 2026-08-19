"""Regression: stop-verify must prefer the project venv for pytest."""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_core_deps_importable_under_active_interpreter():
    """Fail loudly if pytest is not the project env (torch / arcticdb)."""
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
