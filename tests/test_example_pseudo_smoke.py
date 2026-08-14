"""Smoke test for examples/pseudo_cpcv_smoke.py (Phase 9)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pseudo_cpcv_smoke_exits_zero() -> None:
    script = ROOT / "examples" / "pseudo_cpcv_smoke.py"
    assert script.is_file(), "missing examples/pseudo_cpcv_smoke.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "CPCV folds=" in proc.stdout
    assert "paths=" in proc.stdout
