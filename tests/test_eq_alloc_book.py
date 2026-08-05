"""D4: renders the whole eq allocation book from synthetic inputs.

Asserts figure count, manifest completeness, footer stamping on every page,
and fail-closed behaviour when no strategy carries any weight columns.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mascotrl.reporting import eq_alloc_book as book_mod
from mascotrl.reporting.eq_alloc_book import render_eq_alloc_book
from mascotrl.reporting.strategy_persistence import strategy_frame

N = 80
K = 8
SECIDS = [f"S{i:03d}" for i in range(K)]


def _rand_frame(seed: int, *, with_weights: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-06", periods=N)
    if not with_weights:
        return pd.DataFrame(
            {
                "date": dates,
                "turnover": rng.uniform(0, 0.2, N),
                "cost": rng.uniform(0, 0.001, N),
                "gross": rng.normal(0, 0.01, N),
                "total_net": rng.normal(0.0002, 0.01, N),
                "residual": rng.normal(0.0001, 0.008, N),
            }
        )
    w = rng.dirichlet(np.ones(K) * 2.0, size=N)
    return strategy_frame(
        dates=dates,
        secids=SECIDS,
        weights=w,
        turnover=rng.uniform(0, 0.2, N),
        cost=rng.uniform(0, 0.001, N),
        gross=rng.normal(0, 0.01, N),
        total_net=rng.normal(0.0003, 0.01, N),
        residual=rng.normal(0.0001, 0.008, N),
    )


def _synthetic_strategy_frames() -> dict[str, pd.DataFrame]:
    return {
        "policy": _rand_frame(1),
        "equal_weight": _rand_frame(2),
        "min_variance_lw": _rand_frame(3),
        "olps:pamr": _rand_frame(4),
        "ceiling:kelly_cnn": _rand_frame(5),
    }


def _synthetic_results() -> dict:
    rng = np.random.default_rng(0)
    path_pnls = {f"0:{i}": list(rng.normal(0.0002, 0.01, 60)) for i in range(3)}
    return {
        "k": K,
        "pool": 40,
        "wall_total_s": 123.4,
        "breadth": {
            "dii": {"n_eff_enb": 5.2},
            "densest": {"n_eff_enb": 3.1},
        },
        "surface_signals": {"allowlist": []},
        "confirmatory": {
            "estimand_hash": "abc123",
            "benchmark_sharpes": {"equal_weight": 0.4, "min_variance_lw": 0.3},
            "benchmark_sharpes_residual": {"equal_weight": 0.1, "min_variance_lw": 0.05},
            "benchmark_estimand_hashes": {"equal_weight": "abc123", "min_variance_lw": "abc123"},
            "olps_estimand_hashes": {"pamr": "abc123"},
            "ceiling_estimand_hashes": {"kelly_cnn": "abc123"},
            "path_summary": {
                "sharpe_mean": 0.55,
                "sharpe_std": 0.12,
                "n_seeds": 2,
                "per_seed": [0.5, 0.6],
                "path_sharpes": [0.4, 0.5, 0.6, 0.7],
            },
            "fill_ladder": {"mid": 0.6, "pct75": 0.5, "pct95": 0.4},
            "path_pnls": path_pnls,
            "gates": {
                "gate1": {"break_even_spread_multiplier": 0.4, "pass": True, "decision": "continue_positive_framing"},
                "gate2": {"pass": False, "decision": "pivot_negative_economic_framing"},
                "gate3": {"pass": True, "decision": "continue_positive_framing"},
            },
            "stats_table": {
                "estimand_hash": "abc123",
                "deflated_sharpe": {
                    "sharpe_ann": 0.55, "psr": 0.8, "dsr": 0.6, "n_trials": 10, "n_obs": 500,
                    "significant_05": False,
                },
                "hansen_spa_vs_ew": {
                    "ok": True, "n_obs": 500, "n_boot": 199, "block_mean": 21,
                    "t_spa": 1.2, "pvalue_lower": 0.3, "pvalue_consistent": 0.4, "pvalue_upper": 0.5,
                },
                "romano_wolf_vs_ew": {"rejected": ["policy"], "results": [{"name": "policy", "p_value": 0.03}]},
                "cscv_pbo": {"pbo": 0.35, "median_logit": 0.1, "n_paths": 3, "n_obs": 60, "n_partitions_used": 20},
            },
            "negative_controls_prelim": {"shuffled_panel_ew_sharpe": 0.02},
        },
    }


def test_render_book_produces_all_ten_sections_with_footer_on_every_page(tmp_path: Path) -> None:
    frames = _synthetic_strategy_frames()
    results = _synthetic_results()
    ff4 = np.random.default_rng(7).normal(0, 0.01, size=(N, 4))

    real_stamp_footer = book_mod.stamp_footer
    with patch.object(book_mod, "stamp_footer", wraps=real_stamp_footer) as spy:
        index = render_eq_alloc_book(
            strategy_frames=frames,
            out_dir=tmp_path / "book",
            results=results,
            cfg={"k": K},
            ff4_factors=ff4,
            known_limitations=["Universe capped at K=8 for the smoke test."],
        )
    # Every page added to the multi-page book.pdf went through stamp_footer.
    assert spy.call_count == index["n_pages"]

    # All ten sections (0..10 inclusive) contributed at least one figure entry.
    section_ids = {e["id"].split(".")[0] for e in index["figures"]}
    assert section_ids == {f"S{i}" for i in range(11)}
    assert index["n_figures"] >= 30
    assert index["n_written"] >= 20

    # Manifest completeness: every stamp_footer field populated.
    m = index["manifest"]
    for field in ("git_sha", "config_sha", "estimand_hash", "scorecard", "date_start", "date_end"):
        assert m.get(field), f"manifest missing {field}"

    # Every "written" entry's declared files really exist on disk.
    for e in index["figures"]:
        if e["status"] != "written":
            continue
        for key in ("png", "pdf", "csv", "json"):
            if key in e:
                assert Path(e[key]).is_file(), f"{e['id']} declares missing {key}: {e[key]}"

    assert (tmp_path / "book" / "book.pdf").is_file()
    assert (tmp_path / "book" / "BOOK.md").is_file()
    assert (tmp_path / "book" / "index.json").is_file()

    # index.json sha256 map covers every real file under out_dir.
    on_disk = {
        str(p.relative_to(tmp_path / "book"))
        for p in (tmp_path / "book").rglob("*")
        if p.is_file() and p.name != "index.json"
    }
    assert set(index["sha256"].keys()) == on_disk


def test_render_book_fails_closed_when_no_strategy_has_weights(tmp_path: Path) -> None:
    frames = {"policy": _rand_frame(1, with_weights=False)}
    with pytest.raises(ValueError, match="no strategy carries any weight columns"):
        render_eq_alloc_book(strategy_frames=frames, out_dir=tmp_path / "book")


def test_render_book_fails_closed_on_empty_strategy_frames(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no strategy carries any weight columns"):
        render_eq_alloc_book(strategy_frames={}, out_dir=tmp_path / "book")


def test_render_book_degrades_gracefully_with_minimal_inputs(tmp_path: Path) -> None:
    """Only a bare policy frame and no results: most figures skip, none crash."""
    frames = {"policy": _rand_frame(1)}
    index = render_eq_alloc_book(strategy_frames=frames, out_dir=tmp_path / "book2")
    assert index["n_figures"] >= 30
    assert index["n_skipped"] >= 1
    # BOOK.md still narrates every figure, including skipped ones with a note.
    text = (tmp_path / "book2" / "BOOK.md").read_text()
    assert "[skipped]" in text
