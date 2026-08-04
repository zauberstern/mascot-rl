"""Probability of Backtest Overfitting appendix (Bailey et al.) on trial Sharpes.

This is an AFML Ch.12 *motivated* honesty appendix — NOT purged combinatorial
RL retrain paths. Nested WFO in this repo remains expanding fine-tune without
purge/embargo; do not claim CPCV equivalence.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def probability_of_backtest_overfitting(
    trial_sharpes: Sequence[float],
    *,
    n_partitions: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """
    CSCV-style PBO on a vector of trial Sharpes (ablation/plugin/fold trials).

    For each combinatorial split of trials into equal IS/OOS halves (when
    feasible) or random balanced partitions, compute the fraction of splits
    where the IS-best trial underperforms the OOS median (classic PBO idea).

    Parameters
    ----------
    trial_sharpes
        Annualized Sharpes from distinct trials (folds, ablations, seeds).
    n_partitions
        Number of random balanced partitions when C(N, N/2) is huge.
    """
    arr = np.asarray(list(trial_sharpes), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    out: dict[str, Any] = {
        "n_trials": n,
        "protocol": "cscv_trial_sharpes_appendix",
        "citation": (
            "Bailey, Borwein, López de Prado, Zhu — Probability of Backtest "
            "Overfitting (motivating CSCV/PBO); López de Prado AFML Ch.12. "
            "NOT purged combinatorial RL retrain paths."
        ),
        "nested_wfo_is_not_cpcv": True,
        "pbo": float("nan"),
        "logit_pbo": float("nan"),
        "n_partitions_used": 0,
    }
    if n < 4:
        out["reason"] = "need >=4 finite trial Sharpes"
        return out

    half = n // 2
    if half < 2:
        out["reason"] = "need half>=2"
        return out

    rng = np.random.default_rng(int(seed))
    # Cap random partitions; exact combinations explode.
    n_part = int(max(8, min(n_partitions, 200)))
    fails = 0
    used = 0
    for _ in range(n_part):
        idx = rng.permutation(n)
        is_idx = idx[:half]
        oos_idx = idx[half : 2 * half]
        if is_idx.size < 2 or oos_idx.size < 2:
            continue
        is_best = int(is_idx[np.argmax(arr[is_idx])])
        # Relative OOS rank of the IS winner among OOS set (by OOS Sharpe of same trials)
        # Map: we need OOS performance of the same trial ids.
        # Here each "trial" is a scalar Sharpe (already OOS fold or ablation OOS).
        # Classic PBO uses IS/OOS on returns; with only Sharpes we use a proxy:
        # fail if IS-best Sharpe is below median of the held-out trial Sharpes.
        oos_vals = arr[oos_idx]
        is_best_sharpe = float(arr[is_best])
        med = float(np.median(oos_vals))
        if is_best_sharpe < med:
            fails += 1
        used += 1
    if used == 0:
        out["reason"] = "no partitions"
        return out
    pbo = float(fails / used)
    # logit with soft clip
    eps = 1e-6
    p = min(max(pbo, eps), 1.0 - eps)
    out["pbo"] = pbo
    out["logit_pbo"] = float(np.log(p / (1.0 - p)))
    out["n_partitions_used"] = used
    out["interpretation"] = (
        "High PBO → selection among trials often fails to generalize to held-out "
        "trial Sharpes. Low PBO is necessary but not sufficient for skill."
    )
    return out


def append_trial_ledger_entry(
    path: Path | str,
    *,
    source: str,
    trial_id: str,
    sharpe: float | None = None,
    status: str = "ok",
    config_sha: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one auditable trial to ``logs/trial_ledger.json`` (create if missing)."""
    from datetime import datetime, timezone

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    blob: dict[str, Any]
    if p.is_file():
        try:
            blob = json.loads(p.read_text())
        except Exception:
            blob = {"trials": []}
    else:
        blob = {"trials": [], "schema": "mascotrl.trial_ledger.v1"}
    trials = list(blob.get("trials") or [])
    entry: dict[str, Any] = {
        "source": str(source),
        "id": str(trial_id),
        "status": str(status),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if sharpe is not None and np.isfinite(float(sharpe)):
        entry["sharpe"] = float(sharpe)
    if config_sha:
        entry["config_sha"] = str(config_sha)
    if extra:
        entry.update(extra)
    trials.append(entry)
    blob["trials"] = trials
    blob["n_trials_listed"] = len(trials)
    blob["updated_at"] = entry["timestamp"]
    p.write_text(json.dumps(blob, indent=2))
    return blob


def build_trial_ledger(
    *,
    ablation_rows: list[dict[str, Any]] | None = None,
    plugin_rows: list[dict[str, Any]] | None = None,
    nested_fold_sharpes: list[float] | None = None,
    multiseed_sharpes: list[float] | None = None,
) -> dict[str, Any]:
    """Auditable trial list feeding n_trials / PBO."""
    trials: list[dict[str, Any]] = []
    for row in ablation_rows or []:
        if row.get("status") and row.get("status") != "ok":
            continue
        sh = row.get("oos_sharpe")
        if sh is None:
            continue
        trials.append(
            {
                "source": "ablation",
                "id": row.get("id") or row.get("label"),
                "sharpe": float(sh),
            }
        )
    for row in plugin_rows or []:
        if row.get("status") != "ok":
            continue
        sh = row.get("oos_sharpe_net")
        if sh is None:
            continue
        trials.append(
            {
                "source": "plugin",
                "id": row.get("id") or row.get("label"),
                "sharpe": float(sh),
                "evidence_tier": row.get("evidence_tier"),
            }
        )
    for i, sh in enumerate(nested_fold_sharpes or []):
        if sh is None or not np.isfinite(sh):
            continue
        trials.append({"source": "nested_fold", "id": f"fold_{i}", "sharpe": float(sh)})
    for i, sh in enumerate(multiseed_sharpes or []):
        if sh is None or not np.isfinite(sh):
            continue
        trials.append({"source": "multiseed", "id": f"seed_{i}", "sharpe": float(sh)})

    sharpes = [t["sharpe"] for t in trials]
    pbo = probability_of_backtest_overfitting(sharpes)
    return {
        "n_trials_listed": len(trials),
        "trials": trials,
        "pbo_appendix": pbo,
        "note": (
            "Conservative ledger for multiple-testing honesty. Nested WFO ≠ CPCV; "
            "PBO here uses trial Sharpes, not purged combinatorial RL paths."
        ),
    }
