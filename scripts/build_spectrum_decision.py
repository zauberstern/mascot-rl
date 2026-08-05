"""Rank spectrum cells vs reference; emit decision matrix JSON."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mascotrl.eval.spectrum_multiple_testing import (
    n_trials_breakdown,
    paired_mde,
    romano_wolf_stepdown,
)
from mascotrl.spectrum.registry import metric_orientation


def load_cells(art_dir: Path) -> list[dict]:
    cells = []
    for path in sorted(art_dir.glob("*.json")):
        if path.name in ("index.json", "decision_matrix.json", "spectrum_summary.json"):
            continue
        if path.name.endswith("_policy_behavior.json"):
            continue
        if path.name.startswith("timing_probe_"):
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(blob, dict) or "spectrum_cell_id" not in blob:
            continue
        cells.append(blob)
    return cells


def _extract_eval_metric(cell: dict) -> tuple[float, str, str]:
    """Return (metric, claim_metric, orientation) from cell artifact."""
    tr = cell.get("transfer_report") or {}
    claim = str(tr.get("claim_metric") or cell.get("claim_metric") or "sharpe_mean")
    orient = str(
        tr.get("metric_orientation")
        or cell.get("metric_orientation")
        or metric_orientation(claim)
    )
    m = tr.get("eval_metric")
    if m is None or (isinstance(m, float) and not math.isfinite(m)):
        runner = cell.get("runner_artifact") or {}
        path_sum = runner.get("path_summary") or {}
        m = path_sum.get("sharpe_mean")
        if m is None:
            pol = runner.get("policy") or {}
            m = pol.get(claim) or pol.get("cao_y") or pol.get("mean_pl")
    try:
        val = float(m) if m is not None else float("nan")
    except (TypeError, ValueError):
        val = float("nan")
    return val, claim, orient


def beats_reference(
    cell_metric: float,
    ref_metric: float,
    *,
    orientation: str = "higher_better",
) -> bool:
    """Orientation-aware economic beat vs reference cell."""
    if not (cell_metric == cell_metric and ref_metric == ref_metric):
        return False
    if orientation == "lower_better":
        return bool(cell_metric < ref_metric)
    return bool(cell_metric > ref_metric)


def _diffs_vs_reference(cells: list[dict], ref_id: str) -> dict[str, list[float]]:
    """Placeholder paired diffs: use scalar eval_metric gap as length-1 series.

    Real campaigns should stamp per-path diffs under runner_artifact['path_diffs_vs_ref'].
    """
    by_id = {c["spectrum_cell_id"]: c for c in cells}
    ref = by_id.get(ref_id)
    if ref is None:
        return {}
    ref_m, _, orient = _extract_eval_metric(ref)
    out: dict[str, list[float]] = {}
    for c in cells:
        cid = c["spectrum_cell_id"]
        if cid == ref_id:
            continue
        runner = c.get("runner_artifact") or {}
        stamped = runner.get("path_diffs_vs_ref")
        if stamped:
            out[cid] = [float(x) for x in stamped]
            continue
        m, _, cell_orient = _extract_eval_metric(c)
        o = cell_orient or orient
        if not (m == m and ref_m == ref_m):
            continue
        # Orient so positive means cell beats reference.
        if o == "lower_better":
            out[cid] = [float(ref_m - m)]
        else:
            out[cid] = [float(m - ref_m)]
    return out


def _resolve_reference_id(cells: list[dict]) -> str | None:
    """Per-arm reference id (B-REF). Prefer stamped reference, else ``{arm}_reference``."""
    by_id = {c.get("spectrum_cell_id"): c for c in cells}
    if "reference" in by_id:
        return "reference"
    # Prefer explicit is_reference stamp.
    for c in cells:
        if bool(c.get("is_reference")):
            return str(c.get("spectrum_cell_id"))
    arms = {str(c.get("arm") or c.get("portfolio_arm") or "") for c in cells}
    arms.discard("")
    if len(arms) == 1:
        arm = next(iter(arms))
        cand = f"{arm}_reference"
        if cand in by_id:
            return cand
    for arm in ("eq", "opt", "mix"):
        cand = f"{arm}_reference"
        if cand in by_id:
            return cand
    return None


def build_decision(
    cells: list[dict],
    *,
    n_seeds: int = 1,
    n_cost_rungs: int = 1,
    n_boot: int = 200,
    reference_cell_id: str | None = None,
) -> dict:
    by_id = {c["spectrum_cell_id"]: c for c in cells}
    ref_id = reference_cell_id or _resolve_reference_id(cells)
    ref = by_id.get(ref_id) if ref_id else None
    ref_metric = float("nan")
    ref_orient = "higher_better"
    ref_claim = "sharpe_mean"
    if ref is not None:
        ref_metric, ref_claim, ref_orient = _extract_eval_metric(ref)

    diffs = _diffs_vs_reference(cells, ref_id or "reference")
    # B-RW: drop length-1 padding; only run Romano-Wolf when series length >= 2.
    diffs_for_rw = {k: v for k, v in diffs.items() if len(v) >= 2}
    if diffs_for_rw:
        rw = romano_wolf_stepdown(diffs_for_rw, n_boot=int(n_boot), seed=0)
    else:
        rw = {
            "adjusted_pvalues": {},
            "rejected": {},
            "n_boot": 0,
            "protocol": "skipped_insufficient_path_length",
        }
    adj = rw.get("adjusted_pvalues") or {}

    trials = n_trials_breakdown(len(cells), int(n_seeds), int(n_cost_rungs))

    rows = []
    for c in cells:
        tr = c.get("transfer_report") or {}
        cg = c.get("collapse_guard") or {}
        cid = c.get("spectrum_cell_id")
        cell_m, claim, orient = _extract_eval_metric(c)
        beat = False
        if ref_id is not None and cid != ref_id:
            beat = beats_reference(cell_m, ref_metric, orientation=orient or ref_orient)
        p_adj = adj.get(cid)
        rows.append(
            {
                "spectrum_cell_id": cid,
                "spectrum_axis": c.get("spectrum_axis"),
                "arm": c.get("arm"),
                "resolved": c.get("resolved"),
                "promotable": bool(c.get("promotable")),
                "real_reference_arm_present": bool(tr.get("real_reference_arm_present")),
                "collapse_ok": bool(cg.get("ok")),
                "eval_metric": cell_m,
                "claim_metric": claim,
                "metric_orientation": orient,
                "beats_reference": bool(beat),
                "romano_wolf_p_adjusted": p_adj,
                "note": (
                    "beats_reference is orientation-aware vs reference eval_metric; "
                    "promotion also requires RW p_adj<0.05, transfer_report, collapse_ok"
                ),
            }
        )

    # Paired MDE placeholder from residual dispersion of reference if present.
    sigma_d = float("nan")
    n_paths = 0
    if ref is not None:
        cg = ref.get("collapse_guard") or {}
        sigma_d = float(cg.get("residual_vs_bs_dispersion") or float("nan"))
        runner = ref.get("runner_artifact") or {}
        n_paths = int(runner.get("n_paths") or 0)
        if n_paths <= 0 and math.isfinite(sigma_d):
            n_paths = 1
    mde = paired_mde(sigma_d, n_paths) if n_paths > 0 else float("nan")

    return {
        "reference_cell": ref_id,
        "reference_eval_metric": ref_metric,
        "reference_claim_metric": ref_claim,
        "reference_metric_orientation": ref_orient,
        "n_rows": len(rows),
        "rows": rows,
        "n_trials_breakdown": trials,
        "romano_wolf": {
            "adjusted_pvalues": adj,
            "rejected": rw.get("rejected"),
            "n_boot": rw.get("n_boot"),
            "protocol": rw.get("protocol"),
        },
        "paired_mde": {
            "delta_min": mde,
            "sigma_d": sigma_d,
            "n": n_paths,
            "formula": "2.802 * sigma_d / sqrt(n)",
        },
        "promotion_rule": (
            "beats reference AND peers (random/sign_lag/long/BS-delta) at pct75 "
            "under CPCV with transfer_report and collapse_guard.ok; "
            "Romano-Wolf adjusted p < 0.05 across spectrum cells"
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--art-dir",
        type=Path,
        default=ROOT / "logs" / "artifacts" / "spectrum",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--n-cost-rungs", type=int, default=1)
    p.add_argument("--n-boot", type=int, default=200)
    args = p.parse_args()
    out = args.out or (args.art_dir / "decision_matrix.json")
    decision = build_decision(
        load_cells(args.art_dir),
        n_seeds=args.n_seeds,
        n_cost_rungs=args.n_cost_rungs,
        n_boot=args.n_boot,
    )
    out.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out} n_rows={decision['n_rows']}")


if __name__ == "__main__":
    main()
