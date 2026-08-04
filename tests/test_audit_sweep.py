"""Tests for scripts/audit_sweep.py mechanical scanner."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_sweep import PATTERNS, scan_file


def test_scan_detects_or_range_and_nan_to_num(tmp_path: Path) -> None:
    p = tmp_path / "sample.py"
    p.write_text(
        "secids = cfg.get('_universe_secids') or range(int(rets.shape[1]))\n"
        "x = np.nan_to_num(obs, nan=0.0)\n"
        "try:\n"
        "    f()\n"
        "except Exception:\n"
        "    pass\n",
        encoding="utf-8",
    )
    hits = scan_file(p)
    patterns = {h["pattern"] for h in hits}
    assert "or_range" in patterns
    assert "nan_to_num" in patterns
    assert "except_pass_block" in patterns or "except_pass" in patterns


def test_patterns_cover_required_families() -> None:
    names = {n for n, _ in PATTERNS}
    for required in (
        "except_pass",
        "nan_to_num",
        "fillna_zero",
        "as_of_none",
        "or_range",
        "shell_true",
        "campaign_synthetic",
    ):
        assert required in names


def test_audit_sweep_main_runs(tmp_path: Path) -> None:
    from scripts.audit_sweep import main

    out = tmp_path / "hits.jsonl"
    rc = main(["--out", str(out)])
    assert rc == 0
    assert out.is_file()
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines, "expected at least one candidate in the live tree"
    json.loads(lines[0])  # valid JSONL
