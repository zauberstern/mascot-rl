#!/usr/bin/env python3
"""Benchmark CMDP projection wall-clock vs K (cvxpylayers KKT scaling)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mascotrl.policy.convex_projection import ConvexProjectionLayer


def bench_k(
    ks: list[int],
    reps: int = 20,
    *,
    base_aum_usd: float = 50_000_000.0,
) -> list[dict]:
    rows = []
    for k in ks:
        layer = ConvexProjectionLayer(num_assets=k, turnover_limit=0.15)
        w_raw = torch.randn(8, k)
        w_prev = torch.zeros(8, k)
        deltas = torch.randn(8, k)
        # warmup
        for _ in range(3):
            _ = layer(w_raw, w_prev, deltas, vol_scale=0.2)
        t0 = time.perf_counter()
        for _ in range(reps):
            _ = layer(w_raw, w_prev, deltas, vol_scale=0.2)
        dt = (time.perf_counter() - t0) / reps * 1e3
        per_name = float(base_aum_usd) / max(k, 1)
        rows.append(
            {
                "K": k,
                "ms_per_forward_batch8": dt,
                "ms_per_sample": dt / 8.0,
                "base_aum_usd": float(base_aum_usd),
                "notional_per_name_usd": per_name,
                "note": (
                    f"At ${base_aum_usd:,.0f} gross book, K={k} ⇒ "
                    f"~${per_name:,.0f}/name before ADV haircut."
                ),
            }
        )
        print(
            f"K={k:4d}  {dt:8.2f} ms / batch8  ({dt/8:.2f} ms/sample)  "
            f"~${per_name:,.0f}/name @ ${base_aum_usd:,.0f} AUM"
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ks", default="10,25,50,75,100,150")
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--base-aum-usd", type=float, default=50_000_000.0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    rows = bench_k(ks, reps=args.reps, base_aum_usd=args.base_aum_usd)
    break_k = None
    for r in rows:
        if r["ms_per_sample"] > 50.0:
            break_k = r["K"]
            break
    out = {
        "rows": rows,
        "wallclock_break_k_ms_per_sample_gt_50": break_k,
        "production_ceiling_k": 50,
        "base_aum_usd": float(args.base_aum_usd),
        "note": (
            "cvxpylayers exact KKT; production single-book target remains K≤50 "
            f"at illustrative base AUM ${args.base_aum_usd:,.0f}. "
            "Do not scale past K=50 without SCS/ADMM or approx projection."
        ),
    }
    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
