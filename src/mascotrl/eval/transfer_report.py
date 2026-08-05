"""Sim-to-real transfer report (replaces rBergomi-only ban with measured gap)."""
from __future__ import annotations

from typing import Any, Mapping


def build_transfer_report(
    *,
    train_metric: float,
    eval_metric: float,
    train_world: str,
    eval_world: str = "optionmetrics",
    real_reference_arm_present: bool = False,
    real_reference_metric: float | None = None,
    claim_metric: str = "sharpe_mean",
    metric_orientation: str = "higher_better",
) -> dict[str, Any]:
    """Compare train-world claim metric vs OptionMetrics eval under matched costs.

    ``transfer_gap`` = o * (train_metric - eval_metric) where
    ``o = +1`` for higher_better and ``o = -1`` for lower_better.
    Positive gap means train flattered the policy relative to eval.
    Promotion requires ``real_reference_arm_present`` (Mikkila rule).
    """
    orient = str(metric_orientation or "higher_better").lower().strip()
    if orient not in ("higher_better", "lower_better"):
        raise ValueError(
            f"metric_orientation must be 'higher_better' or 'lower_better'; got {metric_orientation!r}"
        )
    o = 1.0 if orient == "higher_better" else -1.0
    t = float(train_metric)
    e = float(eval_metric)
    if t == t and e == e:  # not NaN
        gap = o * (t - e)
        gap_pct = gap / abs(t) if abs(t) > 1e-12 else float("nan")
    else:
        gap = float("nan")
        gap_pct = float("nan")
    out: dict[str, Any] = {
        "claim_metric": str(claim_metric),
        "metric_orientation": orient,
        "train_world": str(train_world),
        "eval_world": str(eval_world),
        "train_metric": t,
        "eval_metric": e,
        "transfer_gap": gap,
        "transfer_gap_pct": gap_pct,
        "real_reference_arm_present": bool(real_reference_arm_present),
    }
    if real_reference_metric is not None:
        out["real_reference_metric"] = float(real_reference_metric)
    return out


def refuse_promotion_without_real_arm(report: Mapping[str, Any]) -> Mapping[str, Any]:
    """Fail closed: no promoted claim without a real-data reference arm."""
    if not bool(report.get("real_reference_arm_present")):
        raise ValueError(
            "real_reference_arm_present is required for promotion "
            "(Mikkila rule: measure empiric path training, not sim alone)"
        )
    return report
