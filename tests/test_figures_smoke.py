"""Smoke tests for spectrum figure MVP (synthetic artifacts, no lake)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _write(path: Path, blob: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob), encoding="utf-8")


def _synth_cpcv(arm: str, *, sharpes: list[float], be: float | None) -> dict:
    rng = np.random.default_rng(abs(hash(arm)) % (2**31))
    paths = []
    for i, sh in enumerate(sharpes):
        n = 40
        pnl = (rng.normal(0.0, 0.01, size=n) + 0.001 * sh).tolist()
        dates = [f"2022-01-{d:02d}" for d in range(1, n + 1)]
        paths.append(
            {
                "path_id": i,
                "dates": dates,
                "pnl": pnl,
                "n_days": n,
                "sharpe": sh,
                "mean_pnl": float(np.mean(pnl)),
            }
        )
    be_val = float("nan") if be is None else float(be)
    return {
        "cpcv_config": {
            "n_splits": 6,
            "n_test_groups": 2,
            "finetune_passes": 3,
            "purge_days": 5,
            "embargo_days": 2,
        },
        "path_summary": {
            "n_paths": len(sharpes),
            "sharpe_mean": float(np.mean(sharpes)),
            "sharpe_median": float(np.median(sharpes)),
            "sharpe_std": float(np.std(sharpes)),
            "sharpe_p05": float(np.percentile(sharpes, 5)),
            "sharpe_p95": float(np.percentile(sharpes, 95)),
            "positive_path_rate": float(np.mean(np.asarray(sharpes) > 0)),
            "path_sharpes": sharpes,
        },
        "paths": paths,
        "gate1": {
            "break_even_spread_multiplier": be_val,
            "min_required": 0.25,
            "pass": False,
            "decision": "pivot_negative_economic_framing",
        },
        "cost_ladder": {"break_even_spread_multiplier": be_val},
        "arm": arm,
    }


def _synth_gate3(rl_sharpe: float) -> dict:
    return {
        "schema": "mascotrl.gate3_same_fold.v1",
        "gate3": {
            "baselines": {
                "equal_weight": {"sharpe": -0.4},
                "quintile_spread": {"sharpe": -0.2},
                "myopic_mv": {"sharpe": -0.3},
                "xgb": {"sharpe": 0.05},
                "mlp": {"sharpe": 0.08},
                "happo_cpcv_mean_path_sharpe": {"sharpe": rl_sharpe},
            },
            "incremental_vs_best_ml": {
                "rl_sharpe": rl_sharpe,
                "ml_sharpe": 0.08,
                "incremental_sharpe": rl_sharpe - 0.08,
                "rl_beats_ml": rl_sharpe > 0.08,
            },
            "decision": "pivot_negative_economic_framing",
        },
    }


def _synth_attrition() -> dict:
    return {
        "screens": {
            "n_base": 10000,
            "fail_iv_present": 1000,
            "fail_volume_positive": 800,
            "fail_mid_above_tick": 500,
            "fail_moneyness_band": 400,
            "fail_no_arbitrage_bounds": 200,
            "fail_standard_settlement": 100,
            "fail_common_stock": 50,
            "fail_not_index_option": 40,
            "fail_no_dividend_in_life": 300,
            "fail_calls_only": 200,
            "fail_spot_missing": 10,
            "n_retained": 6400,
            "n_secids": 50,
        }
    }


def test_figures_smoke_f01_f02_f26(tmp_path: Path):
    arms_root = tmp_path / "artifacts" / "arms"
    flat = tmp_path / "artifacts"
    latex_out = tmp_path / "latex_figs"
    art_out = tmp_path / "art_figs"

    for arm, sharpes, be in (
        ("opt", [-0.5, -0.3, 0.1, -0.4, -0.2], None),
        ("eq", [0.2, 0.3, 0.1, 0.4, 0.15], 0.4),
        ("mix", [-0.1, 0.05, -0.05, 0.0, 0.1], 0.1),
    ):
        _write(arms_root / arm / "cpcv_path_summary.json", _synth_cpcv(arm, sharpes=sharpes, be=be))
        _write(arms_root / arm / "gate3_same_fold.json", _synth_gate3(float(np.mean(sharpes))))

    _write(flat / "filter_attrition.json", _synth_attrition())
    _write(
        flat / "spectrum_summary.json",
        {
            "schema": "mascotrl.spectrum_summary.v1",
            "arms": {
                "opt": {"sharpe_mean": -0.26, "best_industry": -0.2, "best_ml": 0.08, "equal_weight": -0.4},
                "eq": {"sharpe_mean": 0.23, "best_industry": 0.1, "best_ml": 0.12, "equal_weight": 0.05},
                "mix": {"sharpe_mean": 0.0, "best_industry": 0.05, "best_ml": 0.1, "equal_weight": -0.1},
            },
        },
    )

    from src.reporting.figures.core_suite import render_spectrum_figures

    manifest = render_spectrum_figures(
        arms_root,
        latex_out,
        artifacts_flat=flat,
        artifacts_fig_dir=art_out,
        lake_panel=tmp_path / "missing_dh_cross_section.parquet",
    )

    assert isinstance(manifest, dict)
    assert "figures" in manifest
    by_id = {f["id"]: f for f in manifest["figures"]}

    for fig_id in ("F01", "F02", "F26"):
        assert fig_id in by_id, f"missing {fig_id} in manifest"
        entry = by_id[fig_id]
        assert entry.get("status") == "written", entry
        # W5: per-figure PDF off; PNG only (consolidated book.pdf elsewhere).
        assert "pdf" not in entry, entry
        p = Path(entry["png"])
        assert p.is_file(), f"{fig_id} png missing: {p}"
        assert p.stat().st_size > 0
        assert not p.with_suffix(".pdf").is_file(), f"{fig_id} orphan per-figure pdf"

    # Dual write: latex + artifacts copies
    assert (latex_out / "F01_spectrum_sharpe_ladder.png").is_file()
    assert (art_out / "F01_spectrum_sharpe_ladder.png").is_file()
    assert (art_out / "figure_manifest.json").is_file()

    # F16 should skip without lake panel (no crash)
    if "F16" in by_id:
        assert by_id["F16"].get("status") in {"skipped", "written"}
