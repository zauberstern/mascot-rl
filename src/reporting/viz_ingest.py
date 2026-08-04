"""Ingest run metrics JSONL → analysis frames + optional Parquet cache.

Regime convention for this stack (honest adaptation of the tearsheet brief):
  * ``in_sample``  — training episodes (policy learning on synthetic paths)
  * ``out_of_sample`` — held-out eval episodes (frozen policy)

Calendar IS/OOS on CRSP dates does not apply: Layer-1 paths are synthetic
rBergomi surfaces; macro is PIT-clipped but episodes are not a single market
timeline. Visual demarcation uses the regime flag, not a clock date.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


REGIME_IS = "in_sample"
REGIME_OOS = "out_of_sample"

# Stylized linear friction for gross-vs-net charts only (not in training reward).
# Drag ≈ cost_per_unit_turnover * Σ ‖Δw‖₁. Documented as illustrative.
DEFAULT_FRICTION_PER_TURNOVER = 0.02


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_episode_frames(run_dir: Path | str) -> dict[str, pd.DataFrame]:
    """Load train/eval/checkpoint JSONL with regime columns."""
    run_dir = Path(run_dir)
    metrics = run_dir / "metrics"

    train = pd.DataFrame(_read_jsonl(metrics / "episode_train.jsonl"))
    if not train.empty:
        if "regime" not in train.columns:
            train["regime"] = REGIME_IS
        else:
            train["regime"] = train["regime"].fillna(REGIME_IS)
        if "mode" not in train.columns:
            train["mode"] = "happo"
        else:
            train["mode"] = train["mode"].fillna("happo")
        train["source"] = "train"

    eval_df = pd.DataFrame(_read_jsonl(metrics / "episode_eval.jsonl"))
    if not eval_df.empty:
        if "regime" not in eval_df.columns:
            eval_df["regime"] = REGIME_OOS
        else:
            eval_df["regime"] = eval_df["regime"].fillna(REGIME_OOS)
        eval_df["source"] = "eval"

    ckpt = pd.DataFrame(_read_jsonl(metrics / "checkpoints.jsonl"))

    steps = _load_step_samples(metrics)

    return {"train": train, "eval": eval_df, "checkpoints": ckpt, "steps": steps}


def _load_step_samples(metrics_dir: Path) -> pd.DataFrame:
    pq = metrics_dir / "step_samples.parquet"
    jl = metrics_dir / "step_samples.jsonl"
    if pq.is_file():
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    if jl.is_file():
        return pd.DataFrame(_read_jsonl(jl))
    return pd.DataFrame()


def stylized_net_pnl(
    gross_pnl: float,
    mean_turnover: float,
    n_steps: int,
    cost_per_turnover: float = DEFAULT_FRICTION_PER_TURNOVER,
) -> float:
    """Illustrative net after linear turnover drag (not Almgren–Chriss)."""
    drag = float(cost_per_turnover) * float(mean_turnover) * float(max(n_steps, 0))
    return float(gross_pnl) - drag


def enrich_episodes(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    cost_per_turnover: float = DEFAULT_FRICTION_PER_TURNOVER,
    turnover_limit: float = 0.15,
) -> pd.DataFrame:
    """Stack train+eval into one analytics table with NAV helpers."""
    frames = []
    if not train.empty:
        frames.append(train.copy())
    if not eval_df.empty:
        frames.append(eval_df.copy())
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)
    if "pnl" not in df.columns:
        return df

    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    for col in ("mean_turnover", "max_turnover", "mean_abs_delta", "max_abs_delta", "n_steps"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "n_steps" not in df.columns:
        df["n_steps"] = 1
    if "mean_turnover" not in df.columns:
        df["mean_turnover"] = 0.0

    df["gross_pnl"] = df["pnl"]
    df["net_pnl_stylized"] = [
        stylized_net_pnl(
            float(r.gross_pnl),
            float(r.mean_turnover) if pd.notna(r.mean_turnover) else 0.0,
            int(r.n_steps) if pd.notna(r.n_steps) else 0,
            cost_per_turnover,
        )
        for r in df.itertuples()
    ]
    df["turnover_breach"] = np.maximum(
        0.0,
        pd.to_numeric(df.get("max_turnover", 0.0), errors="coerce").fillna(0.0) - turnover_limit,
    )
    # Delta slack proxy: realized |w·Δ| (projection soft-bound magnitude).
    if "max_abs_delta" in df.columns:
        df["delta_slack_proxy"] = pd.to_numeric(df["max_abs_delta"], errors="coerce").fillna(0.0)
    else:
        df["delta_slack_proxy"] = 0.0

    return df


def build_nav_series(
    episodes: pd.DataFrame,
    *,
    mode: str = "happo",
    pnl_col: str = "pnl",
    base: float = 100.0,
) -> pd.DataFrame:
    """
    Compound episodic PnL into a NAV index.

    Episodes are ordered: all in-sample (train) then out-of-sample (eval for mode).
    """
    if episodes.empty:
        return pd.DataFrame(columns=["idx", "regime", "pnl", "nav", "drawdown"])

    is_part = episodes[(episodes["regime"] == REGIME_IS) & (episodes.get("mode", "happo") == mode)]
    oos_part = episodes[(episodes["regime"] == REGIME_OOS) & (episodes["mode"] == mode)]

    # Prefer chronological ep order within each regime
    parts = []
    if not is_part.empty:
        parts.append(is_part.sort_values("ep" if "ep" in is_part.columns else is_part.index))
    if not oos_part.empty:
        parts.append(oos_part.sort_values("ep" if "ep" in oos_part.columns else oos_part.index))
    if not parts:
        # Fallback: any rows matching mode
        sub = episodes[episodes.get("mode", pd.Series(["happo"] * len(episodes))) == mode]
        if sub.empty:
            sub = episodes
        parts = [sub]

    ordered = pd.concat(parts, ignore_index=True)
    pnl = ordered[pnl_col].astype(float).fillna(0.0).to_numpy()
    # Relative increments: treat episode PnL as additive wealth units (not % returns).
    # Scale to return-like increments via wealth_unit so NAV stays interpretable.
    scale = max(np.nanstd(pnl), 1.0)
    rets = pnl / (10.0 * scale)
    nav = base * np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    out = pd.DataFrame(
        {
            "idx": np.arange(len(ordered)),
            "ep": ordered["ep"].to_numpy() if "ep" in ordered.columns else np.arange(len(ordered)),
            "regime": ordered["regime"].to_numpy()
            if "regime" in ordered.columns
            else np.array([REGIME_IS] * len(ordered)),
            "pnl": pnl,
            "ret_proxy": rets,
            "nav": nav,
            "drawdown": dd,
            "source": ordered["source"].to_numpy() if "source" in ordered.columns else "unknown",
        }
    )
    return out


def is_oos_split_index(nav: pd.DataFrame) -> int | None:
    """Index of first OOS point (boundary for shading), or None."""
    if nav.empty or "regime" not in nav.columns:
        return None
    oos = np.where(nav["regime"].to_numpy() == REGIME_OOS)[0]
    return int(oos[0]) if len(oos) else None


def write_metrics_parquet(run_dir: Path | str, frames: Mapping[str, pd.DataFrame] | None = None) -> dict[str, str]:
    """Materialize Parquet under metrics/ for memory-safe downstream viz."""
    run_dir = Path(run_dir)
    metrics = run_dir / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    if frames is None:
        frames = load_episode_frames(run_dir)

    written: dict[str, str] = {}
    for name, df in frames.items():
        if df is None or df.empty:
            continue
        path = metrics / f"{name}_episodes.parquet" if name in ("train", "eval") else metrics / f"{name}.parquet"
        if name == "train":
            path = metrics / "train_episodes.parquet"
        elif name == "eval":
            path = metrics / "eval_episodes.parquet"
        elif name == "checkpoints":
            path = metrics / "checkpoints.parquet"
        elif name == "steps":
            path = metrics / "step_samples.parquet"
        try:
            df.to_parquet(path, index=False)
            written[name] = str(path)
        except Exception:
            # Fallback without pyarrow engine quirks
            alt = path.with_suffix(".csv")
            df.to_csv(alt, index=False)
            written[name] = str(alt)
    return written


def load_report_arrays(run_dir: Path | str) -> dict[str, Any]:
    """Pull optional arrays from overnight_report_full.json if present."""
    run_dir = Path(run_dir)
    full = run_dir / "report" / "overnight_report_full.json"
    compact = run_dir / "report" / "run_report.json"
    if not compact.is_file():
        compact = run_dir / "report" / "overnight_report.json"
    path = full if full.is_file() else compact
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def rolling_sharpe(x: np.ndarray, window: int, ann: float = 252.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size < window or window < 2:
        return np.array([])
    out = np.full(x.size - window + 1, np.nan)
    for i in range(out.size):
        w = x[i : i + window]
        sd = w.std()
        out[i] = (w.mean() / sd * np.sqrt(ann)) if sd > 1e-12 else 0.0
    return out


def rolling_sortino(x: np.ndarray, window: int, ann: float = 252.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size < window or window < 2:
        return np.array([])
    out = np.full(x.size - window + 1, np.nan)
    for i in range(out.size):
        w = x[i : i + window]
        downside = w[w < 0.0]
        dd = downside.std() if downside.size else 0.0
        out[i] = (w.mean() / dd * np.sqrt(ann)) if dd > 1e-12 else (np.inf if w.mean() > 0 else 0.0)
    return out


def herfindahl(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float).ravel()
    s = np.abs(w).sum()
    if s < 1e-12:
        return 0.0
    p = np.abs(w) / s
    return float((p * p).sum())
