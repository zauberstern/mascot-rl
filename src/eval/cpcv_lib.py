"""CPCV via purgedcv (AFML ch. 12), with MascotRL fold/resume contracts.

Uses ``CombinatorialPurgedCV`` for combinatorial test blocks, purge, and
embargo. A synthetic trading-day time axis maps index-based ``purge_days`` /
``embargo_days`` onto purgedcv's calendar horizons so geometry matches the
hand-rolled :mod:`src.eval.cpcv` builder (exact parity under residual-equity
defaults).

Resume / path reconstruction stay on the existing MascotRL helpers so
manifest keys and ``CPCVFold`` metadata are unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from src.eval.cpcv import (
    _CPCV_FOLD_AUX_KEY,
    CPCVConfig,
    CPCVFold,
    _contiguous_windows,
    assign_paths,
    group_bounds,
    reconstruct_paths,
    summarize_paths,
)
from src.eval.campaign_manifest import (
    ManifestFlushWriter,
    get_completed_extra,
    get_completed_pnl,
    is_cell_complete,
    load_manifest,
)
from src.eval.stats_rigor import annualized_sharpe
from src.logging_utils import get_logger

log = get_logger("mascotrl.eval.cpcv_lib")

# Synthetic epoch: one Timedelta day == one trading-day index step.
_SYNTH_EPOCH = pd.Timestamp("2000-01-01")


def _trading_day_times(n: int) -> pd.Series:
    """Monotonic timestamps where Δt=1 day equals one panel trading day."""
    return pd.Series(
        pd.date_range(_SYNTH_EPOCH, periods=int(n), freq="D")
    )


def _fold_test_groups(
    test_idx: np.ndarray, bounds: list[tuple[int, int]]
) -> tuple[int, ...]:
    """Map a test index set to sorted group ids that are fully covered."""
    test_set = {int(i) for i in test_idx}
    groups: list[int] = []
    for g, (lo, hi) in enumerate(bounds):
        block = set(range(lo, hi + 1))
        if block and block.issubset(test_set):
            groups.append(g)
    return tuple(groups)


def build_cpcv_folds_lib(
    dates: Sequence[pd.Timestamp],
    cfg: CPCVConfig | None = None,
    *,
    extra_purge_indices: Sequence[int] | None = None,
    extra_purge_radius: int | None = None,
) -> list[CPCVFold]:
    """Build CPCV folds using purgedcv; return MascotRL ``CPCVFold`` objects.

    Purge/embargo use a synthetic trading-day clock so ``purge_days`` /
    ``embargo_days`` match the hand-rolled index semantics. Evaluation times
    are offset by ``purge_days + 1`` synthetic days so the overlap rule matches
    AFML-style contiguous purge of ``purge_days`` before each test block
    (purgedcv's overlap is half-open relative to our inclusive range).
    """
    from purgedcv import CombinatorialPurgedCV

    cfg = cfg or CPCVConfig()
    cfg.validate()
    dates = list(dates)
    n = len(dates)
    if n < cfg.n_splits:
        raise ValueError(f"need >= {cfg.n_splits} dates, got {n}")

    bounds = group_bounds(dates, cfg.n_splits)
    pred = _trading_day_times(n)
    # +1 matches inclusive purge of purge_days trading days before test start.
    purge_shift = int(cfg.purge_days) + (1 if cfg.purge_days > 0 else 0)
    base = pd.date_range(_SYNTH_EPOCH, periods=n, freq="D")
    evalu = pd.Series(
        [base[min(i + purge_shift, n - 1)] for i in range(n)]
    )
    splitter = CombinatorialPurgedCV(
        n_splits=int(cfg.n_splits),
        n_test_groups=int(cfg.n_test_groups),
        prediction_times=pred,
        evaluation_times=evalu,
        purge_horizon=None,
        embargo_observations=int(cfg.embargo_days) if cfg.embargo_days > 0 else None,
    )

    radius = int(cfg.purge_days if extra_purge_radius is None else extra_purge_radius)
    extra_base = {int(i) for i in (extra_purge_indices or []) if 0 <= int(i) < n}
    extra_expanded: set[int] = set()
    for i in extra_base:
        for j in range(max(0, i - radius), min(n, i + radius + 1)):
            extra_expanded.add(j)

    folds: list[CPCVFold] = []
    X = np.zeros((n, 1), dtype=np.float64)
    for fid, (train_idx, test_idx) in enumerate(splitter.split(X, None)):
        test_idx = np.asarray(test_idx, dtype=int)
        train_idx = np.asarray(train_idx, dtype=int)
        test_set = {int(i) for i in test_idx}
        reselect_purge = {j for j in extra_expanded if j not in test_set}
        if reselect_purge:
            train_idx = np.asarray(
                sorted(int(i) for i in train_idx if int(i) not in reselect_purge),
                dtype=int,
            )
        # Recover purged / embargo counts for diagnostics (same as hand-rolled).
        combo = _fold_test_groups(test_idx, bounds)
        purged: set[int] = set(reselect_purge)
        embargoed: set[int] = set()
        for g in combo:
            lo, hi = bounds[g]
            for j in range(max(0, lo - int(cfg.purge_days)), lo):
                if j not in test_set:
                    purged.add(j)
            for j in range(hi + 1, min(n, hi + 1 + int(cfg.embargo_days))):
                if j not in test_set:
                    embargoed.add(j)
        folds.append(
            CPCVFold(
                fold_id=int(fid),
                test_groups=combo,
                test_windows=_contiguous_windows(dates, np.sort(test_idx)),
                train_windows=_contiguous_windows(dates, train_idx),
                n_test_days=int(test_idx.size),
                n_train_days=int(train_idx.size),
                n_purged_days=len(purged),
                n_embargoed_days=len(embargoed),
            )
        )
    log.info(
        "CPCV(purgedcv): %d folds over %d dates (%d groups, %d test) → %d paths",
        len(folds),
        n,
        cfg.n_splits,
        cfg.n_test_groups,
        cfg.n_paths(),
    )
    return folds


def run_cpcv_lib(
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
    """Same contract as :func:`src.eval.cpcv.run_cpcv`, folds from purgedcv."""
    cfg = cfg or CPCVConfig()
    folds = build_cpcv_folds_lib(
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
            else:
                try:
                    raw = fold_runner(fold) or {}
                except Exception as exc:
                    log.warning("CPCV(purgedcv) fold %d failed: %s", fold.fold_id, exc)
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
        "backend": "purgedcv",
        "reference": "Lopez de Prado, AFML (2018), ch. 7 and 12; purgedcv CombinatorialPurgedCV",
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
        "n_paths_assign": len(assign_paths(cfg)),
    }
