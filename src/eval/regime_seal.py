"""Seal / replay cache for regime scorecard walk-forward HMM.

Eval-only artifact. Not a capital/tradable checkpoint and does not feed the
RL train loop. Freezes a *causal* chronology (labels + per-window HMMs keyed
by train_end). Never promotes a full-sample EM fit.

SCHEMA 3: ``turbulent`` = operational filtered Markov P>0.5;
``turbulent_q75`` = Skulls expanding-q75 comparator.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.walk_forward_hmm import (
    MarkovCheckpoint,
    apply_hmm_checkpoint_filter,
    apply_markov_checkpoint_filter,
)

SCHEMA_VERSION = 3
_SEAL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def sealed_dir(out_root: Path | str, name: str) -> Path:
    if not _SEAL_NAME_RE.match(name):
        raise ValueError(f"invalid seal name: {name!r}")
    return Path(out_root) / "sealed" / name


def _content_hash(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def _git_commit(repo_root: Path | None) -> str | None:
    if repo_root is None:
        return None
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def seal_regime_run(
    *,
    name: str,
    out_root: Path | str,
    scorecard: dict[str, Any],
    series: dict[str, Any],
    models: dict[int, Any] | None,
    asset_returns: np.ndarray,
    hyperparams: dict[str, Any],
    repo_root: Path | str | None = None,
    fioracle_hash: str | None = None,
    hmm_step: int = 21,
) -> Path:
    """Write sealed/<name>/{manifest.json, regime_series.parquet, hmm_windows/}."""
    if (scorecard.get("hygiene") or {}).get("status") == "fail":
        raise ValueError("refusing to seal: hygiene failed")
    agr = scorecard.get("agreement") or {}
    if agr.get("status") == "unavailable":
        raise ValueError("refusing to seal: agreement unavailable")
    fit_h = agr.get("fit_hygiene") or {}
    labeled_frac = fit_h.get("labeled_frac")
    if labeled_frac is not None and float(labeled_frac) < 0.80:
        raise ValueError(
            f"refusing to seal: Markov labeled_frac={labeled_frac} < 0.80"
        )

    dest = sealed_dir(out_root, name)
    if dest.exists():
        raise FileExistsError(f"seal already exists: {dest}")
    dest.mkdir(parents=True, exist_ok=False)
    hmm_dir = dest / "hmm_windows"
    hmm_dir.mkdir()

    dates = series.get("dates")
    if dates is None:
        raise ValueError("series must include dates")
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    turb = np.asarray(series.get("turbulence"), dtype=np.float64)
    turb_mask = np.asarray(series.get("turbulent_mask"), dtype=bool)
    p_high = np.asarray(series.get("hmm_p_highvol"), dtype=np.float64)
    hard = np.asarray(series.get("hmm_hard"), dtype=np.int32)
    hard_piger = series.get("hmm_hard_piger")
    if hard_piger is None:
        hard_piger = np.full(len(idx), -1, dtype=np.int32)
    else:
        hard_piger = np.asarray(hard_piger, dtype=np.int32)
    hard_growing = series.get("hmm_hard_growing")
    if hard_growing is None:
        hard_growing = np.full(len(idx), -1, dtype=np.int32)
    else:
        hard_growing = np.asarray(hard_growing, dtype=np.int32)
    chi2 = series.get("turbulent_chi2")
    if chi2 is None:
        chi2 = np.zeros(len(idx), dtype=bool)
    else:
        chi2 = np.asarray(chi2, dtype=bool)
    q75 = series.get("turbulent_q75")
    if q75 is None:
        q75 = np.zeros(len(idx), dtype=bool)
    else:
        q75 = np.asarray(q75, dtype=bool)
    infl_p = series.get("inflation_p_high")
    if infl_p is None:
        infl_p = np.full(len(idx), np.nan, dtype=np.float64)
    else:
        infl_p = np.asarray(infl_p, dtype=np.float64)

    labels = series.get("labels")
    if labels is None:
        regime = np.full(len(idx), "calm", dtype=object)
    else:
        regime = np.asarray(list(labels), dtype=object)

    # Overlay rule (a): calm→crisis only on operational turbulent days.
    regime_out = regime.copy()
    for i, flag in enumerate(turb_mask):
        if flag and str(regime_out[i]) == "calm":
            regime_out[i] = "crisis"

    frame = pd.DataFrame(
        {
            "turbulence": turb,
            "turbulent": turb_mask,
            "turbulent_q75": q75,
            "turbulent_chi2": chi2,
            "hmm_p_highvol": p_high,
            "hmm_hard": hard,
            "hmm_hard_piger": hard_piger,
            "hmm_hard_growing": hard_growing,
            "regime": regime_out.astype(str),
            "inflation_p_high": infl_p,
        },
        index=idx,
    )
    frame.index.name = "date"
    parquet_path = dest / "regime_series.parquet"
    frame.to_parquet(parquet_path)

    train_ends = [int(t) for t in (series.get("train_ends") or [])]
    saved_ends: list[int] = []
    if models:
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover
            raise ImportError("seal requires joblib") from exc
        for end, model in models.items():
            end_i = int(end)
            joblib.dump(model, hmm_dir / f"end_{end_i}.joblib")
            saved_ends.append(end_i)
        saved_ends.sort()
    else:
        saved_ends = sorted(train_ends)

    root = Path(repo_root) if repo_root is not None else None
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "calendar_start": str(idx.min().date()) if len(idx) else None,
        "calendar_end": str(idx.max().date()) if len(idx) else None,
        "n_rows": int(len(idx)),
        "hyperparams": {
            **hyperparams,
            "hmm_n_components": 2,
            "hmm_covariance_type": "diag",
            "hmm_random_state": 42,
            "hmm_step": int(hmm_step),
            "expanding_quantile": 0.75,
            "switching_variance": True,
            "operational_label": agr.get(
                "operational_label", "markov_filtered_p05"
            ),
            "overlay_mode": "markov",
            "k_regimes": 2,
        },
        "content_hashes": {
            "asset_returns": _content_hash(asset_returns),
            "fioracle_features": fioracle_hash,
        },
        "metrics": {
            "jaccard_turbulence_hmm": agr.get("jaccard_turbulence_hmm"),
            "jaccard_grade": agr.get("jaccard_grade"),
            "jaccard_turbulence_hmm_piger": agr.get("jaccard_turbulence_hmm_piger"),
            "turbulent_day_frac": agr.get("turbulent_day_frac"),
            "mean_turbulent_run_days": agr.get("mean_turbulent_run_days"),
            "mean_turbulent_run_days_q75": agr.get("mean_turbulent_run_days_q75"),
            "hmm_p11_highvol": agr.get("hmm_p11_highvol"),
            "hmm_expected_duration_highvol": agr.get("hmm_expected_duration_highvol"),
            "jaccard_macro_crisis_turbulence": agr.get(
                "jaccard_macro_crisis_turbulence"
            ),
            "jaccard_inflation_rule_vs_markov": agr.get(
                "jaccard_inflation_rule_vs_markov"
            ),
            "taxonomy_disclaimer": agr.get("taxonomy_disclaimer"),
            "piger_daily_note": agr.get("piger_daily_note"),
            "macro_source": agr.get("macro_source"),
            "fit_hygiene": agr.get("fit_hygiene"),
            "kpt_monthly": agr.get("kpt_monthly"),
            "growing_window": agr.get("growing_window"),
            "event_alignment": scorecard.get("event_alignment"),
            "causal_per_regime_sharpe": agr.get("causal_per_regime_sharpe"),
        },
        "train_ends": saved_ends,
        "git_commit": _git_commit(root),
    }
    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return dest


def load_sealed_series(seal_path: Path | str) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(seal_path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(path / "regime_series.parquet")
    return frame, manifest


def load_hmm_window(seal_path: Path | str, train_end: int) -> Any:
    """Load checkpoint; caller must only apply it on [train_end, train_end+step)."""
    import joblib

    path = Path(seal_path) / "hmm_windows" / f"end_{int(train_end)}.joblib"
    if not path.is_file():
        raise FileNotFoundError(path)
    return joblib.load(path)


def apply_sealed_checkpoint(
    seal_path: Path | str,
    *,
    train_end: int,
    features: np.ndarray,
    date_index: int,
    hmm_step: int | None = None,
    train_window: int | None = None,
) -> dict[str, np.ndarray]:
    """Apply one sealed HMM/Markov checkpoint to a single date index."""
    _, manifest = load_sealed_series(seal_path)
    step = int(hmm_step if hmm_step is not None else manifest["hyperparams"]["hmm_step"])
    if int(date_index) < int(train_end) or int(date_index) >= int(train_end) + step:
        raise ValueError(
            f"date_index={date_index} outside checkpoint validity "
            f"[{train_end}, {train_end + step})"
        )
    if int(train_end) > int(date_index):
        raise ValueError(
            f"refusing backward application: train_end={train_end} > t={date_index}"
        )
    model = load_hmm_window(seal_path, train_end)
    if isinstance(model, MarkovCheckpoint):
        return apply_markov_checkpoint_filter(
            model,
            np.asarray(features, dtype=np.float64).reshape(-1),
            date_index=int(date_index),
            hmm_step=step,
        )
    tw = train_window
    if tw is None:
        tw = int(manifest["hyperparams"].get("hmm_window", train_end))
    return apply_hmm_checkpoint_filter(
        model,
        features,
        train_end=int(train_end),
        dates_start=int(date_index),
        dates_end=int(date_index) + 1,
        train_window=tw,
    )


def scorecard_from_seal(
    seal_path: Path | str,
    *,
    base_scorecard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild agreement metrics from sealed parquet (no HMM refit)."""
    frame, manifest = load_sealed_series(seal_path)
    schema = int(manifest.get("schema_version") or 1)
    hard = frame["hmm_hard"].to_numpy(dtype=np.int32)
    valid = hard >= 0
    from src.eval.walk_forward_hmm import jaccard_turbulent
    from src.eval.regime_scorecard import run_duration_stats

    # Schema 3: turbulent = operational; Jaccard uses q75 comparator column.
    if "turbulent_q75" in frame.columns and schema >= 3:
        q75 = frame["turbulent_q75"].to_numpy(dtype=bool)
        operational = frame["turbulent"].to_numpy(dtype=bool)
        jacc = (
            float(jaccard_turbulent(q75[valid], operational[valid]))
            if int(valid.sum()) >= 10
            else float("nan")
        )
        duration = run_duration_stats(operational)
        duration_q75 = run_duration_stats(q75)
    else:
        turb = frame["turbulent"].to_numpy(dtype=bool)
        jacc = (
            float(jaccard_turbulent(turb[valid], hard[valid] == 1))
            if int(valid.sum()) >= 10
            else float("nan")
        )
        duration = run_duration_stats(turb)
        duration_q75 = duration

    seal_metrics = manifest.get("metrics") or {}

    agreement = {
        "status": "ok" if np.isfinite(jacc) else "partial",
        "reason": "from_seal",
        "operational_label": (manifest.get("hyperparams") or {}).get(
            "operational_label", "from_seal"
        ),
        "jaccard_turbulence_hmm": jacc,
        "jaccard_note": "replayed from seal (no HMM refit)",
        "turbulent_day_frac": duration["turbulent_day_frac"],
        "mean_turbulent_run_days": duration["mean_turbulent_run_days"],
        "n_turbulent_runs": duration["n_turbulent_runs"],
        "turbulence_switch_rate": duration["switch_rate"],
        "mean_turbulent_run_days_q75": duration_q75["mean_turbulent_run_days"],
        "n_hmm_labeled": int(valid.sum()),
        "seal_name": manifest.get("name"),
        "seal_metrics": seal_metrics,
        "macro_source": seal_metrics.get("macro_source"),
        "hmm_p11_highvol": seal_metrics.get("hmm_p11_highvol"),
        "hmm_expected_duration_highvol": seal_metrics.get(
            "hmm_expected_duration_highvol"
        ),
        "causal_per_regime_sharpe": seal_metrics.get("causal_per_regime_sharpe"),
        "taxonomy_disclaimer": seal_metrics.get("taxonomy_disclaimer")
        or "binary operational turb vs 3-state macro; not a detector failure",
        "fit_hygiene": seal_metrics.get("fit_hygiene"),
        "kpt_monthly": seal_metrics.get("kpt_monthly"),
        "growing_window": seal_metrics.get("growing_window"),
    }
    if "hmm_hard_piger" in frame.columns:
        piger = frame["hmm_hard_piger"].to_numpy(dtype=np.int32)
        pv = piger >= 0
        if int(pv.sum()) >= 10:
            cmp = (
                frame["turbulent_q75"].to_numpy(dtype=bool)
                if "turbulent_q75" in frame.columns
                else frame["turbulent"].to_numpy(dtype=bool)
            )
            agreement["jaccard_turbulence_hmm_piger"] = float(
                jaccard_turbulent(cmp[pv], piger[pv] == 1)
            )
    if "turbulent_chi2" in frame.columns:
        chi2 = frame["turbulent_chi2"].to_numpy(dtype=bool)
        cmp = (
            frame["turbulent_q75"].to_numpy(dtype=bool)
            if "turbulent_q75" in frame.columns
            else frame["turbulent"].to_numpy(dtype=bool)
        )
        agreement["chi2_turbulent_day_frac"] = float(chi2.mean())
        agreement["jaccard_empirical_vs_chi2"] = float(jaccard_turbulent(cmp, chi2))
    else:
        agreement["chi2_turbulent_day_frac"] = float("nan")
        agreement["jaccard_empirical_vs_chi2"] = float("nan")

    if "regime" in frame.columns:
        crisis = frame["regime"].astype(str).to_numpy() == "crisis"
        op = frame["turbulent"].to_numpy(dtype=bool)
        agreement["jaccard_macro_crisis_turbulence"] = float(
            jaccard_turbulent(crisis, op)
        )
        agreement["jaccard_macro_crisis_note"] = (
            "cross-taxonomy (3-state macro vs binary turbulence); "
            "not a detector failure"
        )

    if isinstance(jacc, float) and np.isfinite(jacc):
        agreement["jaccard_grade"] = "agree" if jacc >= 0.4 else "limitation"

    events = seal_metrics.get("event_alignment")
    if not isinstance(events, dict):
        events = {"status": "unavailable", "reason": "not stored in seal"}
    agreement["calendar_stress_windows"] = events

    out = dict(base_scorecard) if base_scorecard else {}
    out["agreement"] = agreement
    out["event_alignment"] = events
    out["returns_source"] = (manifest.get("hyperparams") or {}).get(
        "returns_source", "from_seal"
    )
    out["status"] = out.get("status") or "ok"
    out["from_seal"] = {
        "name": manifest.get("name"),
        "path": str(seal_path),
        "schema_version": schema,
        "git_commit": manifest.get("git_commit"),
    }
    limitations = list(out.get("limitations") or [])
    limitations = [
        x
        for x in limitations
        if "asset_returns not provided" not in str(x).lower()
        and "agreement unavailable" not in str(x).lower()
    ]
    if schema < 2:
        limitations.append("seal schema v1; rerun for v2 columns")
    elif schema == 2:
        limitations.append(
            "seal schema v2; turbulent column was q75; rerun v3 for operational Markov"
        )
    out["limitations"] = limitations
    return out
