#!/usr/bin/env python3
"""Probe universe capacity for spectrum K axis (WP-S2).

For each candidate K, check whether the PIT-eligible pool is large enough on
enough rebalance dates. Writes ``logs/artifacts/spectrum/universe_capacity.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.spectrum.capacity_probe import probe_universe_capacity


def probe_from_lake(*, requested: tuple[int, ...]) -> dict:
    """Prefer live lake breadth; fall back to measured 480 secid anchor."""
    pool = 480
    try:
        from src.data.paths import LAKE_ROOT
        import duckdb

        db = duckdb.connect()
        # options_panel year parquet if present
        panel = Path(LAKE_ROOT) / "options_panel"
        years = sorted(panel.glob("year=*")) if panel.is_dir() else []
        if years:
            # Count distinct secids from the latest year partition if parquet.
            y = years[-1]
            files = list(y.glob("*.parquet")) + list(y.rglob("*.parquet"))
            if files:
                q = f"SELECT COUNT(DISTINCT secid) FROM read_parquet('{files[0]}')"
                pool = int(db.execute(q).fetchone()[0])
    except Exception as exc:  # noqa: BLE001
        return {
            **probe_universe_capacity(pool, requested=requested).to_dict(),
            "pool_source": "fallback_480",
            "pool_error": str(exc)[:200],
        }
    res = probe_universe_capacity(pool, requested=requested)
    out = res.to_dict()
    out["pool_source"] = "lake_or_anchor"
    # Ambition ladder for K_max when 400 refused.
    if 400 in res.refused_k:
        ladder = (350, 300, 250, 200)
        for cand in ladder:
            if cand <= pool:
                out["k_max"] = cand
                axis = sorted(set(list(res.feasible_k) + [cand]))
                out["k_axis"] = [k for k in (100, 200, cand) if k in axis or k <= cand]
                # unique sorted
                out["k_axis"] = sorted(set(out["k_axis"]))
                out["note"] = (
                    f"refused_k=[400]: pool_size={pool}; promoted k_max={cand} "
                    f"from ladder {list(ladder)}"
                )
                break
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--requested", default="100,200,400")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "logs" / "artifacts" / "spectrum" / "universe_capacity.json",
    )
    args = p.parse_args(argv)
    requested = tuple(int(x) for x in str(args.requested).split(",") if x.strip())
    payload = probe_from_lake(requested=requested)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} k_axis={payload.get('k_axis')} k_max={payload.get('k_max')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
