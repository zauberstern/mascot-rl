#!/usr/bin/env python3
"""Offline rehydrate of DESKORG solo peer behaviour (no retrain).

Rebuilds ``*_policy_behavior.json`` for the PICK single PPO mean_std_cao peer
using stored weights + sleeve/macro/returns context. Fail-closed if weights
are missing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_spectrum_campaign import refresh_behavior_exports


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pick-dir",
        type=Path,
        default=ROOT / "logs/artifacts/spectrum/cherrypick",
    )
    ap.add_argument(
        "--stem",
        default="eq_K100_single_ppo_mlp_softmax_mean_std_cao",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "logs/artifacts/spectrum/cherrypick_deskorg",
    )
    args = ap.parse_args()
    src = args.pick_dir / f"{args.stem}.json"
    if not src.is_file():
        print(f"refuse: missing peer artifact {src}", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dest = args.out_dir / src.name
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    # Copy companion weights-bearing behaviour if present (refresh overwrites).
    for suffix in ("_policy_behavior.json",):
        peer = args.pick_dir / f"{args.stem}{suffix}"
        if peer.is_file():
            (args.out_dir / peer.name).write_text(
                peer.read_text(encoding="utf-8"), encoding="utf-8"
            )
    summary = refresh_behavior_exports(
        args.out_dir,
        config_dir=ROOT / "config" / "spectrum" / "cherrypick",
    )
    print(json.dumps(summary, indent=2, default=str))
    refreshed_raw = summary.get("refreshed") or []
    beh = args.out_dir / f"{args.stem}_policy_behavior.json"
    if not beh.is_file() and not refreshed_raw:
        print(f"refuse: rehydrate failed for {args.stem}", file=sys.stderr)
        return 3
    print(f"OK peer rehydrated under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
