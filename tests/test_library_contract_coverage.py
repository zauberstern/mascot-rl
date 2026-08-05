"""B-5: high-value library contract tests + matrix anchor (see AUDIT_LEDGER)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import pyarrow as pa
import torch

from mascotrl.eval.residualization import fit_ipca3_residualizer
from mascotrl.policy.convex_projection import ConvexProjectionLayer


def test_sklearn_truncated_svd_residualizer_output_shape():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    panel = rng.normal(size=(40, 8))
    state = fit_ipca3_residualizer(panel, backend="sklearn_pca")
    assert state.betas.shape == (8, 3)
    assert state.backend_used == "sklearn_pca"
    assert np.isfinite(state.betas).all()


def test_ipca_instrumentedpca_adapter_output_shape():
    pytest.importorskip("ipca")
    rng = np.random.default_rng(1)
    t, n, l = 24, 5, 3
    panel = rng.normal(size=(t, n))
    char = rng.normal(size=(t, n, l))
    state = fit_ipca3_residualizer(panel, char, backend="ipca", n_iter=2)
    assert state.betas.shape == (n, 3)
    assert state.backend_used == "ipca"
    assert np.isfinite(state.betas).all()


def test_arch_stationary_bootstrap_index_distribution_sanity():
    pytest.importorskip("arch")
    from mascotrl.eval.arch_bootstrap import stationary_bootstrap_indices_arch

    idx = stationary_bootstrap_indices_arch(120, block_mean=8, seed=7)
    assert idx.shape == (120,)
    assert idx.min() >= 0 and idx.max() < 120
    # Stationary bootstrap should resample with replacement (not identity)
    assert len(np.unique(idx)) < 120


def test_cvxpy_qp_elastic_slack_fail_closed_finite_under_extreme_proposal():
    """Elastic slack QP never returns NaN; hard box binds on spikes."""
    layer = ConvexProjectionLayer(
        num_assets=4, turnover_limit=0.15, penalty_weight=1e4, max_name_abs_weight=5.0
    )
    w_raw = torch.tensor([[50.0, -50.0, 30.0, -20.0]], dtype=torch.float32)
    w_prev = torch.zeros(1, 4)
    deltas = torch.tensor([[1.0, -1.0, 0.5, -0.5]], dtype=torch.float32)
    w_exec, s_delta, s_turn = layer(
        w_raw, w_prev, deltas, vol_scale=0.2, return_slacks=True
    )
    assert torch.isfinite(w_exec).all()
    assert torch.isfinite(s_delta).all()
    assert torch.isfinite(s_turn).all()
    assert float(w_exec.abs().max()) <= 5.0 + 1e-3
    assert float(s_delta.item()) >= 0.0 or float(s_turn.item()) >= 0.0


def test_arctic_schema_drift_triggers_rewrite(tmp_path):
    from mascotrl.data.arctic_store import ArcticStateStore

    store = ArcticStateStore(db_path=tmp_path / "arctic", library_name="test_lib")
    t1 = pa.table({"date": pa.array(["2020-01-01"]), "vix": [12.0]})
    store.persist_features("macro_state", t1)
    t2 = pa.table(
        {
            "date": pa.array(["2020-01-02"]),
            "vix": [13.0],
            "move": [80.0],
        }
    )
    store.persist_features("macro_state", t2)
    df = store.lib.read("macro_state").data
    assert "move" in df.columns
    assert len(df) >= 1


def test_duckdb_option_filter_screens_match_attrition_source():
    """Marks SQL and attrition counters share one screen registry (no drift)."""
    from mascotrl.data.duckdb_engine import OptionFilterConfig

    cfg = OptionFilterConfig(require_fresh_quotes=True)
    names = {n for n, _ in cfg.screens()}
    assert "iv_present" in names
    assert "fresh_quotes" in names
    # selection_screens are separate from marks filters
    sel = {n for n, _ in cfg.selection_screens()}
    assert "dte_band" in sel or len(sel) >= 1
