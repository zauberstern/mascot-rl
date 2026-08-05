"""Tests for institutional tearsheet ingest + rendering."""
from __future__ import annotations

import pytest
from tests.conftest import FLOAT_TOL

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mascotrl.reporting.institutional_tearsheet import render_tearsheet
from mascotrl.reporting.viz_ingest import (
    REGIME_IS,
    REGIME_OOS,
    build_nav_series,
    enrich_episodes,
    herfindahl,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_herfindahl_and_nav():
    assert abs(herfindahl([0.5, 0.5]) - 0.5) < 1e-9
    assert herfindahl([0, 0, 0]) == pytest.approx(0.0, **FLOAT_TOL)
    train = pd.DataFrame(
        {
            "ep": np.arange(30),
            "regime": [REGIME_IS] * 30,
            "mode": ["happo"] * 30,
            "source": ["train"] * 30,
            "pnl": np.linspace(-1, 1, 30),
            "mean_turnover": [0.15] * 30,
            "max_turnover": [0.15] * 30,
            "mean_abs_delta": [1e-5] * 30,
            "max_abs_delta": [1e-4] * 30,
            "n_steps": [100] * 30,
        }
    )
    ev = pd.DataFrame(
        {
            "ep": np.arange(10),
            "regime": [REGIME_OOS] * 10,
            "mode": ["happo"] * 10,
            "source": ["eval"] * 10,
            "pnl": np.linspace(0.1, 0.5, 10),
            "mean_turnover": [0.15] * 10,
            "max_turnover": [0.16] * 10,
            "mean_abs_delta": [1e-5] * 10,
            "max_abs_delta": [1e-4] * 10,
            "n_steps": [100] * 10,
        }
    )
    ep = enrich_episodes(train, ev)
    nav = build_nav_series(ep, mode="happo")
    assert len(nav) == 40
    assert (nav["regime"].iloc[:30] == REGIME_IS).all()
    assert (nav["regime"].iloc[30:] == REGIME_OOS).all()
    assert "net_pnl_stylized" in ep.columns


def test_render_tearsheet_smoke(tmp_path: Path):
    run = tmp_path / "run"
    metrics = run / "metrics"
    report = run / "report"
    metrics.mkdir(parents=True)
    report.mkdir(parents=True)

    rng = np.random.default_rng(0)
    train_rows = []
    for ep in range(60):
        train_rows.append(
            {
                "ep": ep,
                "regime": "in_sample",
                "mode": "happo",
                "pnl": float(rng.normal(0.1, 1.0)),
                "mean_turnover": 0.15,
                "max_turnover": 0.151,
                "mean_abs_delta": 1e-5,
                "max_abs_delta": 1e-4,
                "n_steps": 80,
                "weight_l1_mean": 1.2,
                "train_policy_loss": float(1.0 / (ep + 1)),
                "train_value_loss": float(2.0 / (ep + 1)),
            }
        )
    eval_rows = []
    for ep in range(20):
        for mode, mu in (("happo", 0.4), ("zero", 0.0), ("random", -0.1)):
            eval_rows.append(
                {
                    "ep": ep,
                    "regime": "out_of_sample",
                    "mode": mode,
                    "pnl": float(rng.normal(mu, 0.8)),
                    "mean_turnover": 0.15 if mode != "zero" else 0.0,
                    "max_turnover": 0.15 if mode != "zero" else 0.0,
                    "mean_abs_delta": 1e-5 if mode != "zero" else 0.0,
                    "max_abs_delta": 1e-4 if mode != "zero" else 0.0,
                    "n_steps": 80,
                    "weight_l1_mean": 1.1,
                }
            )
    _write_jsonl(metrics / "episode_train.jsonl", train_rows)
    _write_jsonl(metrics / "episode_eval.jsonl", eval_rows)

    # Minimal step samples for allocation / case study
    steps = []
    for step in range(40):
        rec = {
            "ep": 0,
            "regime": "out_of_sample",
            "mode": "happo",
            "step": step,
            "reward": float(rng.normal(0, 0.1)),
            "cum_pnl": float(step) * 0.01,
            "delta": 1e-5,
            "turnover": 0.15,
            "weight_l1": 1.0 + 0.01 * step,
            "spot_mean": 100.0 * (1.0 + 0.001 * step),
            "atm_vol": 0.2,
        }
        for j in range(6):
            rec[f"w_{j}"] = float(rng.normal(0, 0.2))
        steps.append(rec)
    _write_jsonl(metrics / "step_samples.jsonl", steps)

    (report / "run_report_full.json").write_text(
        json.dumps(
            {
                "policy_losses": list(np.linspace(1.0, 0.1, 60)),
                "value_losses": list(np.linspace(2.0, 0.2, 60)),
                "macro_sample": np.random.randn(80, 3).tolist(),
            }
        ),
        encoding="utf-8",
    )

    meta = render_tearsheet(run, write_parquet=True)
    assert meta["n_plots_files"] >= 10
    assert (run / "report" / "tearsheet" / "index.json").is_file()
    assert (run / "metrics" / "train_episodes.parquet").is_file() or (
        run / "metrics" / "train_episodes.csv"
    ).is_file()
    # Core figures exist as PNG; no PDF by default
    assert (run / "report" / "tearsheet" / "01_equity_curve.png").is_file()
    assert not (run / "report" / "tearsheet" / "16_alpha_spread.png").is_file()
    assert not (run / "report" / "tearsheet" / "10_baseline_alpha.png").is_file()
    assert not (run / "report" / "tearsheet" / "21_eval_boxplot.png").is_file()
    assert not (run / "report" / "tearsheet" / "01_equity_curve.pdf").is_file()
    from PIL import Image

    im = Image.open(run / "report" / "tearsheet" / "11_turnover_adherence.png")
    w, h = im.size
    assert w > h, f"turnover plot should be landscape, got {w}x{h}"
    assert h < 2000, f"turnover plot unexpectedly tall: {w}x{h}"
