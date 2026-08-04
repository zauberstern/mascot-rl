"""K>50 HAPPO must skip cvxpylayers SCS and use ADMM (deadline / ceiling gate)."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch

from src.plugins.admm_projection import ADMMProjectionLayer
from src.plugins.registry import build_happo_engine
from src.policy.convex_projection import ConvexProjectionLayer
from src.reporting.capital_gates import PROJECTION_K_CEILING
from src.spectrum.cell_schema import validate_cell_cfg
from src.spectrum.yaml_loader import load_cell_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_build_happo_engine_k100_uses_admm_not_cvxpy(monkeypatch):
    constructed = {"cvx": 0}
    real_init = ConvexProjectionLayer.__init__

    def wrapped(self, *a, **k):
        constructed["cvx"] += 1
        return real_init(self, *a, **k)

    monkeypatch.setattr(ConvexProjectionLayer, "__init__", wrapped)
    eng = build_happo_engine(100, 32, 8, cfg={"turnover_limit": 0.25})
    assert constructed["cvx"] == 0
    assert isinstance(eng.convex_projection, ADMMProjectionLayer)
    assert eng.use_projection is True
    assert 100 > PROJECTION_K_CEILING
    assert getattr(eng, "_projection_backend", None) == "admm"


def test_build_happo_engine_k50_keeps_cvxpy():
    eng = build_happo_engine(50, 16, 8, cfg={"turnover_limit": 0.15})
    assert isinstance(eng.convex_projection, ConvexProjectionLayer)
    assert getattr(eng, "_projection_backend", "cvxpy") == "cvxpy"


def test_happo_k100_admm_act_finite_and_fast():
    eng = build_happo_engine(100, 32, 8, cfg={"turnover_limit": 0.25})
    w_prev = torch.zeros(1, 100)
    enr = torch.randn(1, 100, 32)
    mac = torch.randn(1, 8)
    deltas = torch.randn(1, 100)
    t0 = time.perf_counter()
    w, lp, v, raw = eng.act_stochastic(enr, mac, w_prev, deltas, vol_scale=0.2)
    dt_ms = (time.perf_counter() - t0) * 1000
    assert torch.isfinite(w).all()
    assert w.shape == (1, 100)
    assert dt_ms < 250.0


def test_convex_projection_passes_scs_max_iters():
    layer = ConvexProjectionLayer(num_assets=4, turnover_limit=0.15)
    captured: dict = {}

    def fake_cvx(*args, **kwargs):
        captured.update(kwargs)
        B = args[0].shape[0]
        return args[0], torch.zeros(B), torch.zeros(B)

    # nn.Module blocks non-Module assignment; bypass via object.__setattr__.
    object.__setattr__(layer, "cvx_layer", fake_cvx)
    w = torch.randn(2, 4)
    layer(w, torch.zeros_like(w), torch.randn(2, 4), vol_scale=0.2)
    sa = captured["solver_args"]
    assert sa["max_iters"] == 250
    assert sa.get("verbose") in (False, 0, None)


def test_happo_progress_line_format():
    from src.eval.research_happo_cpcv import happo_progress_line

    s = happo_progress_line(
        "cpcv_train_start", seed=0, n_splits=3, n_test_groups=1
    )
    assert s.startswith("phase=cpcv_train start")
    assert "seed=0" in s
    t = happo_progress_line(
        "train_step", seed=0, ep=0, n_episodes=12, step=50, max_steps=1600
    )
    assert t.startswith("phase=happo_train")
    assert "step=50" in t


def test_narrative_happo_yaml_schema_and_budget():
    cfg = load_cell_yaml(
        ROOT
        / "config/spectrum/cherrypick/rc6_narrative/eq_K100_multi_happo_mlp_mean_std_cao.yaml"
    )
    validate_cell_cfg(cfg)
    assert cfg["train_env_steps"] == 30000
    assert cfg["seeds"] == [0, 1, 2, 3, 4]
    assert cfg["cpcv_n_splits"] == 3
    assert cfg["cpcv_n_test_groups"] == 1
    assert cfg["protocol_tier"] == "narrative"
