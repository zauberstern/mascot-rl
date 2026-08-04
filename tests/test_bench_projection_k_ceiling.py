"""bench_projection_k restored to live scripts/."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_bench_projection_k_script_exists() -> None:
    assert (ROOT / "scripts" / "bench_projection_k.py").is_file()


def test_bench_projection_k_smoke() -> None:
    import importlib.util

    path = ROOT / "scripts" / "bench_projection_k.py"
    spec = importlib.util.spec_from_file_location("bench_projection_k", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = mod.bench_k([5, 10], reps=2)
    assert len(rows) == 2
    assert rows[0]["K"] == 5
    assert "ms_per_sample" in rows[0]
