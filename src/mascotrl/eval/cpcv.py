"""
Combinatorial Purged Cross-Validation for the option panel.

Implements the protocol of Lopez de Prado, *Advances in Financial Machine
Learning* (Wiley, 2018), chapters 7 and 12:

  * partition the trading calendar into ``n_splits`` contiguous groups;
  * take every combination of ``n_test_groups`` groups as a test set, giving
    ``C(n_splits, n_test_groups)`` folds;
  * **purge** training observations whose label horizon overlaps a test group;
  * **embargo** a buffer immediately after each test group;
  * reassemble the folds into ``C(n-1, k-1)`` time-ordered backtest *paths*.

Why this replaces a single walk-forward: one historical path yields one Sharpe
with no sampling distribution, so robustness and path dependence cannot be
assessed, and the deflated Sharpe has no credible trial variance to work with.
CPCV yields a distribution of out-of-sample paths from the same data.

Scope note for the paper: CPCV controls leakage and parameter overfitting on a
single strategy definition. It does **not** correct for meta-overfitting across
many strategy ideas; that is what the trial ledger and the deflated Sharpe are
for.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from mascotrl.eval.campaign_manifest import (
    ManifestFlushWriter,
    get_completed_extra,
    get_completed_pnl,
    is_cell_complete,
    load_manifest,
)

# fold_runner may stash OOS aux (weights/cost/...) under this key; it is stripped
# before Sharpe path reconstruction and cached on the resume manifest.
_CPCV_FOLD_AUX_KEY = "__aux__"
from mascotrl.eval.stats_rigor import annualized_sharpe
from mascotrl.logging_utils import get_logger

log = get_logger("mascotrl.eval.cpcv")


@dataclass(frozen=True)
class CPCVConfig:
    """
    CPCV geometry.

    Defaults (6 groups, 2 test groups) give 15 folds and 5 reconstructed paths,
    which is the standard worked configuration in AFML ch. 12.
    """

    n_splits: int = 6
    n_test_groups: int = 2
    # Label horizon in trading days. The delta-hedged label spans one session,
    # so a 1-day purge is sufficient; widen if the label horizon widens.
    purge_days: int = 21
    embargo_days: int = 21

    def n_folds(self) -> int:
        return comb(int(self.n_splits), int(self.n_test_groups))

    def n_paths(self) -> int:
        return comb(int(self.n_splits) - 1, int(self.n_test_groups) - 1)

    def validate(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not 1 <= self.n_test_groups < self.n_splits:
            raise ValueError("n_test_groups must satisfy 1 <= k < n_splits")
        if self.purge_days < 0 or self.embargo_days < 0:
            raise ValueError("purge_days and embargo_days must be non-negative")


def residual_equity_cpcv_config() -> CPCVConfig:
    """Research-locked CPCV(6,2); purge/embargo 21 (ultra8x-refine G4 + AFML)."""
    cfg = CPCVConfig(
        n_splits=6,
        n_test_groups=2,
        purge_days=21,
        embargo_days=21,
    )
    cfg.validate()
    return cfg


@dataclass
class CPCVFold:
    fold_id: int
    test_groups: tuple[int, ...]
    test_windows: list[dict[str, str]]
    train_windows: list[dict[str, str]]
    n_test_days: int
    n_train_days: int
    n_purged_days: int
    n_embargoed_days: int


def group_bounds(dates: Sequence[pd.Timestamp], n_splits: int) -> list[tuple[int, int]]:
    """Contiguous, near-equal index ranges ``[lo, hi]`` inclusive."""
    n = len(dates)
    if n < n_splits:
        raise ValueError(f"need >= {n_splits} dates, got {n}")
    edges = np.linspace(0, n, int(n_splits) + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1]) - 1) for i in range(int(n_splits))]


def _contiguous_windows(
    dates: Sequence[pd.Timestamp], idx: np.ndarray
) -> list[dict[str, str]]:
    """Collapse a sorted index set into contiguous [start, end] date windows."""
    if idx.size == 0:
        return []
    out: list[dict[str, str]] = []
    run_start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            out.append(
                {
                    "start": str(pd.Timestamp(dates[run_start]).date()),
                    "end": str(pd.Timestamp(dates[prev]).date()),
                }
            )
            run_start = i
        prev = i
    out.append(
        {
            "start": str(pd.Timestamp(dates[run_start]).date()),
            "end": str(pd.Timestamp(dates[prev]).date()),
        }
    )
    return out


def build_cpcv_folds(
    dates: Sequence[pd.Timestamp],
    cfg: CPCVConfig | None = None,
    *,
    extra_purge_indices: Sequence[int] | None = None,
    extra_purge_radius: int | None = None,
) -> list[CPCVFold]:
    """
    Enumerate purged, embargoed combinatorial folds over a date index.

    Purge removes training days within ``purge_days`` *before* a test block
    (their forward label reaches into the test window); embargo removes
    training days within ``embargo_days`` *after* it (serial correlation).

    ``extra_purge_indices`` (e.g. CRUCIBLE reselect days) are also removed from
    every fold's training set, along with a radius of ``extra_purge_radius``
    (defaults to ``cfg.purge_days``) on each side. The fold field
    ``n_purged_days`` includes these extras; callers may also read
    ``n_purged_at_reselect`` from the returned metadata via
    :func:`stamp_reselect_purge_meta`.
    """
    cfg = cfg or CPCVConfig()
    cfg.validate()
    dates = list(dates)
    bounds = group_bounds(dates, cfg.n_splits)
    n = len(dates)
    radius = int(cfg.purge_days if extra_purge_radius is None else extra_purge_radius)
    extra_base = {int(i) for i in (extra_purge_indices or []) if 0 <= int(i) < n}
    extra_expanded: set[int] = set()
    for i in extra_base:
        for j in range(max(0, i - radius), min(n, i + radius + 1)):
            extra_expanded.add(j)
    folds: list[CPCVFold] = []
    for fid, combo in enumerate(
        combinations(range(cfg.n_splits), cfg.n_test_groups)
    ):
        test_idx = np.concatenate(
            [np.arange(bounds[g][0], bounds[g][1] + 1) for g in combo]
        )
        test_set = set(int(i) for i in test_idx)
        purged: set[int] = set()
        embargoed: set[int] = set()
        for g in combo:
            lo, hi = bounds[g]
            for j in range(max(0, lo - int(cfg.purge_days)), lo):
                if j not in test_set:
                    purged.add(j)
            for j in range(hi + 1, min(n, hi + 1 + int(cfg.embargo_days))):
                if j not in test_set:
                    embargoed.add(j)
        # CRUCIBLE reselect stamps: purge from train even if not near a test block
        reselect_purge = {j for j in extra_expanded if j not in test_set}
        purged |= reselect_purge
        excluded = test_set | purged | embargoed
        train_idx = np.array(
            sorted(i for i in range(n) if i not in excluded), dtype=int
        )
        folds.append(
            CPCVFold(
                fold_id=fid,
                test_groups=tuple(int(g) for g in combo),
                test_windows=_contiguous_windows(dates, np.sort(test_idx)),
                train_windows=_contiguous_windows(dates, train_idx),
                n_test_days=int(test_idx.size),
                n_train_days=int(train_idx.size),
                n_purged_days=len(purged),
                n_embargoed_days=len(embargoed),
            )
        )
    log.info(
        "CPCV: %d folds over %d dates (%d groups, %d test groups) → %d paths"
        "%s",
        len(folds),
        n,
        cfg.n_splits,
        cfg.n_test_groups,
        cfg.n_paths(),
        f"; n_purged_at_reselect={len(extra_expanded)}" if extra_expanded else "",
    )
    return folds


def stamp_reselect_purge_meta(
    dates: Sequence[pd.Timestamp],
    reselect_mask: np.ndarray | Sequence[bool],
    *,
    purge_radius: int = 21,
) -> dict[str, Any]:
    """Diagnostics for CRUCIBLE reselect → CPCV purge wiring."""
    mask = np.asarray(reselect_mask, dtype=bool).reshape(-1)
    n = len(dates)
    if mask.size != n:
        raise ValueError(f"reselect_mask length {mask.size} != n_dates={n}")
    indices = [int(i) for i, v in enumerate(mask) if bool(v)]
    expanded: set[int] = set()
    for i in indices:
        for j in range(max(0, i - int(purge_radius)), min(n, i + int(purge_radius) + 1)):
            expanded.add(j)
    return {
        "n_reselect_days": len(indices),
        "n_purged_at_reselect": len(expanded),
        "reselect_indices": indices,
        "purge_radius": int(purge_radius),
    }

def assign_paths(cfg: CPCVConfig | None = None) -> list[list[tuple[int, int]]]:
    """
    Map folds to backtest paths.

    Each group is tested in exactly ``C(n-1, k-1)`` folds, so the folds can be
    dealt into that many paths where each path covers every group exactly once.
    Returns, per path, a list of ``(group, fold_id)`` in group order.
    """
    cfg = cfg or CPCVConfig()
    cfg.validate()
    combos = list(combinations(range(cfg.n_splits), cfg.n_test_groups))
    # For each group, the folds that test it (stable order).
    by_group: dict[int, list[int]] = {g: [] for g in range(cfg.n_splits)}
    for fid, combo in enumerate(combos):
        for g in combo:
            by_group[g].append(fid)
    n_paths = cfg.n_paths()
    paths: list[list[tuple[int, int]]] = []
    for p in range(n_paths):
        paths.append([(g, by_group[g][p]) for g in range(cfg.n_splits)])
    return paths


def reconstruct_paths(
    dates: Sequence[pd.Timestamp],
    folds: Sequence[CPCVFold],
    fold_pnl: dict[int, dict[str, float]],
    cfg: CPCVConfig | None = None,
    *,
    periods: float | int = 252,
) -> list[dict[str, Any]]:
    """
    Assemble per-fold daily P&L into time-ordered out-of-sample paths.

    ``fold_pnl`` maps ``fold_id -> {date_str: pnl}``. Each path takes each
    group's P&L from the fold assigned to it, so a path is a complete,
    non-overlapping walk across the whole sample by a model that never trained
    on the segment it is scored on.
    """
    cfg = cfg or CPCVConfig()
    dates = list(dates)
    bounds = group_bounds(dates, cfg.n_splits)
    out: list[dict[str, Any]] = []
    for p, assignment in enumerate(assign_paths(cfg)):
        series: list[tuple[str, float]] = []
        missing = 0
        for g, fid in assignment:
            lo, hi = bounds[g]
            for i in range(lo, hi + 1):
                ds = str(pd.Timestamp(dates[i]).date())
                val = (fold_pnl.get(fid) or {}).get(ds)
                if val is None:
                    missing += 1
                    continue
                series.append((ds, float(val)))
        series.sort(key=lambda kv: kv[0])
        pnl = np.asarray([v for _, v in series], dtype=np.float64)
        out.append(
            {
                "path_id": p,
                "dates": [d for d, _ in series],
                "pnl": pnl.tolist(),
                "n_days": int(pnl.size),
                "n_missing_days": int(missing),
                "sharpe": annualized_sharpe(pnl, periods=periods),
                "mean_pnl": float(pnl.mean()) if pnl.size else float("nan"),
            }
        )
    return out


def summarize_paths(paths: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Distribution of path Sharpes — the headline CPCV statistic."""
    sh = np.asarray(
        [p.get("sharpe", float("nan")) for p in paths], dtype=np.float64
    )
    fin = sh[np.isfinite(sh)]
    if fin.size == 0:
        return {
            "n_paths": len(paths),
            "sharpe_mean": float("nan"),
            "sharpe_std": float("nan"),
            "sharpe_p05": float("nan"),
            "sharpe_median": float("nan"),
            "sharpe_p95": float("nan"),
            "positive_path_rate": float("nan"),
            "path_sharpes": sh.tolist(),
        }
    return {
        "n_paths": len(paths),
        "sharpe_mean": float(fin.mean()),
        "sharpe_std": float(fin.std(ddof=0)),
        "sharpe_p05": float(np.percentile(fin, 5)),
        "sharpe_median": float(np.median(fin)),
        "sharpe_p95": float(np.percentile(fin, 95)),
        "positive_path_rate": float((fin > 0).mean()),
        "path_sharpes": sh.tolist(),
    }


# @lat: [[eval#Protocols]]
def run_cpcv(
    dates: Sequence[pd.Timestamp],
    fold_runner: Callable[[CPCVFold], dict[str, float]],
    cfg: CPCVConfig | None = None,
    *,
    resume: bool = True,
    out_dir: str | Path | None = None,
    seed: int = 0,
    arm: str = "default",
    periods: float | int = 252,
    extra_purge_indices: Sequence[int] | None = None,
    extra_purge_radius: int | None = None,
) -> dict[str, Any]:
    """
    Execute CPCV given a callable that trains/evaluates one fold.

    ``fold_runner(fold)`` must train only on ``fold.train_windows`` and return
    ``{date_str: pnl}`` covering ``fold.test_windows``. Injecting the training
    step keeps this module agnostic to the policy and lets it reuse the
    existing per-fold fine-tune machinery.

    When ``resume`` is True and ``out_dir`` is set, completed cells keyed by
    ``(fold_id, seed, arm)`` in ``out_dir/campaign_manifest.json`` are skipped
    and their cached PnL reused. Manifest writes are atomic (temp + rename).

    ``extra_purge_indices`` (CRUCIBLE reselect days) are purged from every
    fold's training set (see :func:`build_cpcv_folds`).
    """
    cfg = cfg or CPCVConfig()
    folds = build_cpcv_folds(
        dates,
        cfg,
        extra_purge_indices=extra_purge_indices,
        extra_purge_radius=extra_purge_radius,
    )
    fold_pnl: dict[int, dict[str, float]] = {}
    fold_aux: dict[int, Any] = {}
    fold_rows: list[dict[str, Any]] = []
    arm_s = str(arm)
    seed_i = int(seed)
    manifest: dict[str, Any] | None = None
    if resume and out_dir is not None:
        manifest = load_manifest(out_dir)

    skipped = 0
    failed_fold_ids: list[int] = []
    failure_reasons: dict[str, str] = {}
    last_fold_id = int(folds[-1].fold_id) if folds else -1
    manifest_writer: ManifestFlushWriter | None = None
    if resume and out_dir is not None:
        manifest_writer = ManifestFlushWriter(out_dir)
    try:
        for fold in folds:
            reused = False
            aux: Any = None
            if (
                resume
                and manifest is not None
                and is_cell_complete(manifest, fold.fold_id, seed_i, arm_s)
            ):
                cached = get_completed_pnl(manifest, fold.fold_id, seed_i, arm_s)
                pnl = dict(cached or {})
                extra = get_completed_extra(manifest, fold.fold_id, seed_i, arm_s) or {}
                aux = extra.get("oos_records")
                reused = True
                skipped += 1
                log.info(
                    "CPCV fold %d skipped (resume cache seed=%d arm=%s)",
                    fold.fold_id,
                    seed_i,
                    arm_s,
                )
            else:
                try:
                    raw = fold_runner(fold) or {}
                except Exception as exc:
                    # A10: record the reason and do NOT mark complete — an empty
                    # pnl must never poison the resume cache (vacuous skip on retry).
                    log.warning("CPCV fold %d failed: %s", fold.fold_id, exc)
                    failed_fold_ids.append(int(fold.fold_id))
                    failure_reasons[str(fold.fold_id)] = str(exc)[:300]
                    pnl = {}
                    aux = None
                else:
                    raw = dict(raw)
                    aux = raw.pop(_CPCV_FOLD_AUX_KEY, None)
                    pnl = {str(k): float(v) for k, v in raw.items()}
                    if resume and out_dir is not None and manifest_writer is not None:
                        if manifest is None:
                            manifest = load_manifest(out_dir)
                        extra = {"oos_records": aux} if aux is not None else None
                        manifest_writer.mark_and_maybe_flush(
                            manifest,
                            fold_id=fold.fold_id,
                            last_fold_id=last_fold_id,
                            seed=seed_i,
                            arm=arm_s,
                            pnl=pnl,
                            extra=extra,
                        )

            fold_pnl[fold.fold_id] = {str(k): float(v) for k, v in pnl.items()}
            if aux is not None:
                fold_aux[int(fold.fold_id)] = aux
            arr = np.asarray(list(fold_pnl[fold.fold_id].values()), dtype=np.float64)
            fold_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "test_groups": list(fold.test_groups),
                    "test_windows": fold.test_windows,
                    "n_test_days": fold.n_test_days,
                    "n_train_days": fold.n_train_days,
                    "n_purged_days": fold.n_purged_days,
                    "n_embargoed_days": fold.n_embargoed_days,
                    "n_pnl_days": int(arr.size),
                    "sharpe": annualized_sharpe(arr, periods=periods),
                    "resumed_from_manifest": bool(reused),
                    "oos_aux_cached": bool(aux is not None),
                }
            )
    finally:
        if manifest_writer is not None and manifest is not None:
            manifest_writer.flush(manifest)

    paths = reconstruct_paths(dates, folds, fold_pnl, cfg, periods=periods)
    return {
        "protocol": "combinatorial_purged_cv",
        "reference": "Lopez de Prado, AFML (2018), ch. 7 and 12",
        "config": {
            "n_splits": cfg.n_splits,
            "n_test_groups": cfg.n_test_groups,
            "purge_days": cfg.purge_days,
            "embargo_days": cfg.embargo_days,
        },
        "n_folds": len(folds),
        "n_failed_folds": len(failed_fold_ids),
        "failed_fold_ids": failed_fold_ids,
        "failure_reasons": failure_reasons,
        "folds": fold_rows,
        "fold_aux": fold_aux,
        "paths": paths,
        "path_summary": summarize_paths(paths),
        "resume": {
            "enabled": bool(resume and out_dir is not None),
            "out_dir": str(out_dir) if out_dir is not None else None,
            "seed": seed_i,
            "arm": arm_s,
            "n_skipped": int(skipped),
        },
        "scope_note": (
            "CPCV addresses leakage and parameter overfitting for one strategy "
            "definition; it does not correct meta-overfitting across strategy "
            "ideas (see trial ledger and deflated Sharpe)."
        ),
    }
