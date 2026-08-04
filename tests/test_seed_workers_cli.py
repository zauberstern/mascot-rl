"""MASCOTRL_SEED_WORKERS / --seed-workers CLI and seed-pack worker roundtrip."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


def test_argparse_accepts_seed_workers() -> None:
    from scripts.run_eq_alloc_campaign import build_arg_parser

    ns = build_arg_parser().parse_args(["--seed-workers", "3"])
    assert int(ns.seed_workers) == 3


def test_argparse_seed_workers_rejects_below_one() -> None:
    from scripts.run_eq_alloc_campaign import build_arg_parser

    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--seed-workers", "0"])


def test_argparse_seed_workers_default_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MASCOTRL_SEED_WORKERS", "2")
    from scripts.run_eq_alloc_campaign import build_arg_parser

    ns = build_arg_parser().parse_args([])
    assert int(ns.seed_workers) == 2


def test_seed_pack_roundtrip_and_worker_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import run_eq_alloc_campaign as camp

    pack_dir = tmp_path / "_seed_pack"
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    t, k = 12, 3
    panel = np.linspace(-0.01, 0.01, t * k, dtype=np.float64).reshape(t, k)
    fac = np.zeros((t, 4), dtype=np.float64)
    dates = pd.date_range("2020-01-01", periods=t, freq="B")
    cfg = {
        "headline_fill": "pct75",
        "projection_mode": "soft",
        "architecture": "mlp",
        "train_epochs": 1,
        "feature_extras": {
            "dollar_volume": np.ones((t, k), dtype=np.float64),
            "include_iv_surface": False,
        },
        "_slot_valid_mask": np.ones((t, k), dtype=bool),
        "_rebalance_mask": np.ones(t, dtype=bool),
        "cpcv_n_splits": 4,
        "cpcv_n_test_groups": 1,
    }
    from src.eval.cpcv import CPCVConfig

    cpcv = CPCVConfig(n_splits=4, n_test_groups=1, purge_days=1, embargo_days=1)
    run_config_hash = "deadbeefcafe0001"
    realized_k = k

    payload_base = camp._write_seed_pack(
        pack_dir,
        panel=panel,
        factors=fac,
        dates=dates,
        cfg=cfg,
        cpcv=cpcv,
        run_config_hash=run_config_hash,
        realized_k=realized_k,
    )
    assert (pack_dir / "panel.npy").is_file()
    assert (pack_dir / "factors.npy").is_file()
    assert (pack_dir / "seed_pack_meta.json").is_file()
    assert (pack_dir / "cfg_runtime.json").is_file()

    meta = json.loads((pack_dir / "seed_pack_meta.json").read_text(encoding="utf-8"))
    assert meta["run_config_hash"] == run_config_hash
    assert meta["realized_k"] == realized_k
    assert len(meta["dates"]) == t

    calls: list[int] = []

    def _fake_cpcv(dates_arg, returns, factors, cfg_arg, **kwargs):
        calls.append(int(kwargs.get("seed", -1)))
        assert np.asarray(returns).shape == (t, k)
        assert np.asarray(factors).shape == (t, 4)
        extras = cfg_arg.get("feature_extras") or {}
        assert isinstance(extras.get("dollar_volume"), np.ndarray)
        assert extras["dollar_volume"].shape == (t, k)
        assert isinstance(cfg_arg.get("_slot_valid_mask"), np.ndarray)
        return {
            "path_summary": {"sharpe_mean": 0.1, "path_sharpes": [0.1]},
            "paths": {"0": {"pnl": {"2020-01-02": 0.01}, "dates": ["2020-01-02"]}},
            "estimand_hash": "abc",
        }

    monkeypatch.setenv("MASCOTRL_THREADS_PER_WORKER", "2")
    with patch("src.eval.research_alpha_cpcv.run_research_alpha_cpcv", _fake_cpcv):
        payload = {
            **payload_base,
            "seed": 7,
            "out_dir": str(out_dir),
            "repo_root": str(camp.ROOT),
            "threads_per": 2,
        }
        art = camp._seed_worker_main(payload)

    assert calls == [7]
    assert art["seed"] == 7
    art_path = out_dir / "cpcv_seed_7.json"
    assert art_path.is_file()
    loaded = json.loads(art_path.read_text(encoding="utf-8"))
    assert loaded["seed"] == 7
    assert (out_dir / "campaign_manifest.json").is_file()
