#!/usr/bin/env python3
"""Mechanical silent-fallback scanner for the Total Correctness audit loop.

Prints candidate findings as JSONL. Does NOT judge Class; ledger triage does.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]

SCAN_GLOBS = (
    "scripts/run_*campaign*.py",
    "scripts/run_research_alpha_cpcv.py",
    "scripts/assign_behavior_codenames.py",
    "scripts/export_figure_blocks.py",
    "scripts/campaign_sprint_finalize.sh",
    "scripts/aws_pull_artifacts.py",
    "scripts/aws_submit_wave.py",
    "src/mascotrl/eval/**/*.py",
    "src/mascotrl/data/**/*.py",
    "src/mascotrl/features/**/*.py",
    "src/mascotrl/reporting/**/*.py",
)

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("except_pass", re.compile(r"except\s+Exception.*:\s*(?:pass|continue)\b")),
    ("nan_to_num", re.compile(r"nan_to_num\s*\(")),
    ("fillna_zero", re.compile(r"fillna\s*\(\s*0")),
    ("np_zeros_fallback", re.compile(r"np\.zeros\s*\(")),
    ("as_of_none", re.compile(r"as_of\s*=\s*None")),
    ("discarded_secids", re.compile(r",\s*_secids\s*=")),
    ("or_range", re.compile(r"or\s+range\s*\(")),
    ("or_empty_dict", re.compile(r"or\s+\{\s*\}")),
    ("shell_true", re.compile(r"\|\|\s*true\b")),
    ("campaign_synthetic", re.compile(r"MASCOTRL_FIGURE_SYNTHETIC")),
    ("setdefault_lake", re.compile(r"""setdefault\s*\(\s*['\"]_?lake_root['\"]""")),
]


def _iter_files() -> Iterator[Path]:
    seen: set[Path] = set()
    for glob in SCAN_GLOBS:
        for p in ROOT.glob(glob):
            if p.is_file() and p not in seen:
                seen.add(p)
                yield p


def scan_file(path: Path) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        try:
            rel = str(path.relative_to(ROOT))
        except ValueError:
            rel = str(path)
        return [
            {
                "pattern": "read_error",
                "path": rel,
                "line": 0,
                "snippet": str(exc)[:120],
            }
        ]
    try:
        rel = str(path.relative_to(ROOT))
    except ValueError:
        rel = str(path)
    out: list[dict] = []
    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        for name, rx in PATTERNS:
            if rx.search(line):
                out.append(
                    {
                        "pattern": name,
                        "path": rel,
                        "line": i,
                        "snippet": line.strip()[:160],
                    }
                )
    # Multiline: except Exception followed within 3 lines by pass/continue
    for i, line in enumerate(lines):
        if re.search(r"except\s+(Exception|BaseException|\w+Error)", line):
            window = "\n".join(lines[i : i + 4])
            if re.search(r"^\s*(pass|continue)\s*$", window, re.M):
                # Avoid double-count if already matched except_pass on same line
                if not re.search(r"except\s+Exception.*:\s*(?:pass|continue)\b", line):
                    out.append(
                        {
                            "pattern": "except_pass_block",
                            "path": rel,
                            "line": i + 1,
                            "snippet": lines[i].strip()[:160],
                        }
                    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", action="store_true", help="Emit JSONL (default)")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output path (default stdout)",
    )
    args = ap.parse_args(argv)
    findings: list[dict] = []
    for path in sorted(_iter_files(), key=lambda p: str(p)):
        findings.extend(scan_file(path))
    text = "".join(json.dumps(f, sort_keys=True) + "\n" for f in findings)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    print(
        f"# audit_sweep: {len(findings)} candidates across "
        f"{len(list(_iter_files()))} files",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
