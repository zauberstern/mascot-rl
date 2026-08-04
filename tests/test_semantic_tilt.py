"""Semantic book-embedding tilt (interpretation only; static text asof)."""
from __future__ import annotations

import math
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from tests.conftest import FLOAT_TOL

ROOT = Path(__file__).resolve().parents[1]


def test_semantic_tilt_metrics_hand_computed() -> None:
    from src.eval.semantic_tilt import semantic_tilt_metrics

    E = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    W = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    e_bar = W @ E
    assert np.allclose(e_bar[0], [1.0, 0.0])
    assert np.allclose(e_bar[1], [0.0, 1.0])
    out = semantic_tilt_metrics(W, E)
    assert out["semantic_rotation_rate"] == pytest.approx(1.0, **FLOAT_TOL)
    assert "semantic_pc1_mean" in out
    assert "semantic_pc2_mean" in out
    assert "semantic_pc3_mean" in out
    assert np.isfinite(out["semantic_pc1_mean"])
    assert np.isfinite(out["semantic_pc2_mean"])


def test_load_firm_text_map_fallback_chain(tmp_path: Path, monkeypatch) -> None:
    from src.eval import semantic_tilt as st

    df = pd.DataFrame(
        {
            "secid": ["101", "202", "303"],
            "TR.BusinessSummary": ["aaa corp summary", "", ""],
            "TR.CommonName": ["Alpha", "Beta Co", ""],
        }
    )
    path = tmp_path / "lseg_ric_map.parquet"
    df.to_parquet(path)
    lake = tmp_path
    (lake / "macro").mkdir(exist_ok=True)
    path.rename(lake / "macro" / "lseg_ric_map.parquet")

    blob = st.load_firm_text_map(lake)
    texts = blob["texts"]
    assert texts["101"].startswith("aaa")
    assert texts["202"] == "Beta Co"
    assert texts.get("303", "") == ""
    assert blob["asof"] == "2026-08-18"
    assert blob["pit"] is False


def test_embed_descriptions_sklearn_fallback() -> None:
    from src.eval.semantic_tilt import _embed_tfidf_svd

    texts = ["alpha banking firm", "beta software cloud", "gamma energy utility"]
    mat = _embed_tfidf_svd(texts)
    assert mat.shape[0] == 3
    assert mat.shape[1] >= 2
    assert np.isfinite(mat).all()
    assert not np.allclose(mat, 0.0)


def test_semantic_tilt_source_not_in_train_cube() -> None:
    src = (ROOT / "src" / "eval" / "semantic_tilt.py").read_text(encoding="utf-8")
    assert "interpretation only" in src.lower()
    if "feeds_capital_gates" in src:
        assert "feeds_capital_gates = False" in src or "feeds_capital_gates: False" in src


def test_nan_when_empty_embeddings() -> None:
    from src.eval.semantic_tilt import semantic_tilt_metrics

    W = np.full((4, 3), 1.0 / 3.0)
    out = semantic_tilt_metrics(W, np.zeros((0, 8)))
    for key in (
        "semantic_rotation_rate",
        "semantic_pc1_mean",
        "semantic_pc2_mean",
        "semantic_pc3_mean",
    ):
        assert math.isnan(out[key])
    assert out.get("data_availability_reason")
