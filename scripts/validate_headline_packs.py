#!/usr/bin/env python3
"""Validate HEAD-EQ pack and launch HEAD-SURF-OFF if missing (WP-S10)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HEAD_EQ_DIR = ROOT / "logs" / "artifacts" / "eq_alloc"
SURF_OFF_DIR = ROOT / "logs" / "artifacts" / "eq_alloc" / "ablation_surface_off"


def validate_head_eq(path: Path) -> dict[str, Any]:
    required = ("results.json", "stats_table.json")
    missing = [n for n in required if not (path / n).is_file()]
    # Also accept campaign summary aliases.
    if missing and (path / "campaign_results.json").is_file():
        missing = [n for n in missing if n != "results.json"]
    return {
        "ok": not missing,
        "path": str(path),
        "missing": missing,
    }


def surf_off_present(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.json"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head-eq-dir", type=Path, default=HEAD_EQ_DIR)
    p.add_argument("--surf-off-dir", type=Path, default=SURF_OFF_DIR)
    p.add_argument("--run-surf-off", action="store_true")
    p.add_argument("--dry-check", action="store_true")
    args = p.parse_args(argv)

    head = validate_head_eq(args.head_eq_dir)
    surf = {"ok": surf_off_present(args.surf_off_dir), "path": str(args.surf_off_dir)}
    report = {"head_eq": head, "head_surf_off": surf}
    out = ROOT / "logs" / "artifacts" / "eq_alloc" / "headline_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

    if args.run_surf_off and not surf["ok"] and not args.dry_check:
        cmd = [
            str(ROOT / ".venv" / "bin" / "python"),
            str(ROOT / "scripts" / "run_eq_alloc_campaign.py"),
            "--no-surface-signals",
            "--out-dir",
            str(args.surf_off_dir),
            "--k",
            "100",
        ]
        print("launching:", " ".join(cmd))
        return subprocess.call(cmd)
    if not head["ok"] or not surf["ok"]:
        print(
            "HEAD pack incomplete; H2/Results freeze blocked until HEAD-EQ validates "
            "and HEAD-SURF-OFF exists. Re-run with --run-surf-off on AWS wave H0."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
