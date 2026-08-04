#!/usr/bin/env python3
"""Wilcoxon signed-rank on matched twin ΔL1 (Scenario A rescue Fix7 / F6).

Reads landed_panel.json twins and tests H1: delta_l1 > 0 (sparse L1 exceeds
softmax L1). Writes A3_matched_twins.json beside the panel.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def twin_wilcoxon(twins: list[dict[str, Any]], *, heads_n: int = 0) -> dict[str, Any]:
    deltas = []
    rows = []
    for t in twins:
        d = t.get("delta_l1")
        try:
            v = float(d)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v):
            continue
        deltas.append(v)
        rows.append(
            {
                "base": t.get("base"),
                "sparse_stem": t.get("sparse_stem"),
                "softmax_stem": t.get("softmax_stem"),
                "sparse_l1": t.get("sparse_l1"),
                "softmax_l1": t.get("softmax_l1"),
                "delta_l1": v,
            }
        )
    arr = np.asarray(deltas, dtype=np.float64)
    out: dict[str, Any] = {
        "n_twins": int(arr.size),
        "delta_l1": {
            "min": float(arr.min()) if arr.size else None,
            "max": float(arr.max()) if arr.size else None,
            "median": float(np.median(arr)) if arr.size else None,
            "mean": float(arr.mean()) if arr.size else None,
            "iqr": float(np.subtract(*np.percentile(arr, [75, 25]))) if arr.size else None,
            "q25": float(np.percentile(arr, 25)) if arr.size else None,
            "q75": float(np.percentile(arr, 75)) if arr.size else None,
        },
        "twins": rows,
        "h0_statement": "delta_l1 > 0 (sparse leaves EW more than matched softmax)",
        "heads_foils_present": heads_n >= 9,
        "heads_foils_note": (
            "RC6_HEADS 9/9 entmax foils landed on panel."
            if heads_n >= 9
            else (
                "RC6_HEADS entmax/Tsallis/tilt-gain foils absent on S3; "
                "F6 supported on landed twins only until HEADS lands."
            )
        ),
    }
    if arr.size < 3:
        out["wilcoxon"] = {
            "statistic": None,
            "pvalue_one_sided_greater": None,
            "significant_01": False,
            "note": "too few twins",
        }
        return out

    nonzero = arr[arr != 0.0]
    if nonzero.size < 3:
        out["wilcoxon"] = {
            "statistic": None,
            "pvalue_one_sided_greater": None,
            "significant_01": False,
            "note": "too many zero deltas",
            "n_nonzero": int(nonzero.size),
        }
        return out

    stat, p = stats.wilcoxon(nonzero, alternative="greater", zero_method="wilcox")
    out["wilcoxon"] = {
        "statistic": float(stat),
        "pvalue_one_sided_greater": float(p),
        "significant_01": bool(p < 0.01),
        "significant_05": bool(p < 0.05),
        "n_nonzero": int(nonzero.size),
        "alternative": "greater",
    }
    out["f6_status"] = (
        "supported_on_landed_twins"
        if p < 0.01
        else "partial_twins_ns"
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--panel",
        type=Path,
        required=True,
        help="Path to landed_panel.json",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output A3_matched_twins.json (default: beside panel)",
    )
    args = ap.parse_args()
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    twins = panel.get("twins") or []
    by_wave = panel.get("by_wave") or {}
    heads_n = int(by_wave.get("RC6_HEADS") or 0)
    # Coverage count: raw wave dir may exceed deduped panel when stems overlap RC6.
    heads_landed = heads_n
    if isinstance(panel.get("by_wave_raw"), dict):
        heads_landed = int(panel["by_wave_raw"].get("RC6_HEADS") or heads_n)
    report = twin_wilcoxon(twins, heads_n=heads_landed)
    report["source_panel"] = str(args.panel)
    report["panel_utc"] = panel.get("utc")
    out = args.out or (args.panel.parent / "A3_matched_twins.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n_twins", "delta_l1", "wilcoxon", "f6_status") if k in report}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
