#!/usr/bin/env python3
"""Write desk_gap_diagnosis.json from sealed regime_desk_series.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.desk_gap_diagnosis import diagnose_desk_payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--desk", type=Path, required=True)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "logs" / "artifacts" / "regime_desk" / "desk_gap_diagnosis.json",
    )
    args = ap.parse_args(argv)
    if not args.desk.is_file():
        print(f"[diagnose_desk_gap] missing desk JSON: {args.desk}", file=sys.stderr)
        return 2
    payload = json.loads(args.desk.read_text(encoding="utf-8"))
    out = diagnose_desk_payload(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "wrote": str(args.out),
                "n_switches": out["n_switches"],
                "ftl_sharpe": out["ftl_sharpe"],
                "oracle_days_by_expert": out["oracle_days_by_expert"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
