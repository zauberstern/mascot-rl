"""Dual-source H.15 + optional EPU/GPRI B3 tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval.regime_dual_source import load_h15_term_oas, resolve_macro_yt_cols
from src.reporting import behavior_metrics as bm


def test_h15_lag1_value_equals_yesterday(tmp_path: Path) -> None:
    usb = tmp_path / "macro"
    usb.mkdir(parents=True)
    dates = pd.bdate_range("2020-01-01", periods=20)
    raw = pd.DataFrame(
        {
            "date": dates,
            "t10y2y": np.arange(20, dtype=float),
            "bamlh0a0hym2": np.arange(20, dtype=float) * 0.1,
        }
    )
    raw.to_parquet(usb / "interest_rate.parquet")
    # Parent of macro/ is usb_root
    out = load_h15_term_oas(tmp_path, dates)
    assert out["source"]["term"] == "h15"
    assert out["term_spread"] is not None
    # Lag-1: day i equals raw day i-1
    assert float(out["term_spread"].iloc[3]) == pytest.approx(2.0)
    assert float(out["hy_oas"].iloc[5]) == pytest.approx(0.4)


def test_resolve_macro_never_averages_sources(tmp_path: Path) -> None:
    dates = pd.bdate_range("2020-01-01", periods=30)
    macro = pd.DataFrame(
        {
            "vix_level": np.full(30, 15.0),
            "hy_oas_level": np.full(30, 4.0),
            # term missing → H.15 fallback
        },
        index=dates,
    )
    usb = tmp_path / "macro"
    usb.mkdir()
    pd.DataFrame(
        {
            "date": dates,
            "t10y2y": np.linspace(0.5, 1.5, 30),
            "bamlh0a0hym2": np.full(30, 3.0),
        }
    ).to_parquet(usb / "interest_rate.parquet")
    cols, sources = resolve_macro_yt_cols(macro, dates, usb_root=tmp_path)
    assert cols is not None
    assert sources["vix"] == "fioracle"
    assert sources["hy_oas"] == "fioracle"
    assert sources["term"] == "h15"
    # Single string per key (never a mix list)
    assert isinstance(sources["term"], str)
    assert sources["term"] in ("fioracle", "h15")


def test_macro_tilt_unchanged_when_epu_none() -> None:
    rng = np.random.default_rng(0)
    T, K = 200, 4
    S = np.eye(K, 7)
    W = np.full((T, K), 0.25)
    vix = rng.standard_normal(T) * 0.2
    hy = rng.standard_normal(T) * 0.05
    term = rng.standard_normal(T) * 0.05
    a = bm.macro_tilt_sensitivity(
        W, sleeve_matrix=S, vix_z=vix, hy_oas_z=hy, term_spread=term
    )
    b = bm.macro_tilt_sensitivity(
        W,
        sleeve_matrix=S,
        vix_z=vix,
        hy_oas_z=hy,
        term_spread=term,
        epu_z=None,
        gpri_z=None,
    )
    assert a["defensive"]["vix_z"]["coef"] == pytest.approx(
        b["defensive"]["vix_z"]["coef"]
    )


def test_macro_tilt_recovers_planted_epu_beta() -> None:
    rng = np.random.default_rng(11)
    T, K = 600, 4
    S = np.zeros((K, 7))
    S[0, 3] = 1.0  # defensive
    S[1, 0] = 1.0
    S[2, 1] = 1.0
    S[3, 2] = 1.0
    vix = rng.standard_normal(T) * 0.05
    hy = rng.standard_normal(T) * 0.05
    term = rng.standard_normal(T) * 0.05
    epu = np.clip(rng.standard_normal(T), -1.5, 1.5) * 0.4
    b_true = 0.25
    tilt = np.zeros(T)
    tilt[1:] = b_true * epu[:-1] + 0.02 * vix[:-1] + rng.normal(0, 0.005, T - 1)
    W = np.full((T, K), 0.25)
    w0 = np.clip(tilt + 0.25, 0.05, 0.95)
    W[:, 0] = w0
    rem = 1.0 - w0
    W[:, 1:] = (rem / 3.0)[:, None]
    sens = bm.macro_tilt_sensitivity(
        W,
        sleeve_matrix=S,
        vix_z=vix,
        hy_oas_z=hy,
        term_spread=term,
        epu_z=epu,
        lags=21,
    )
    assert "epu_z" in sens["defensive"]
    assert sens["defensive"]["epu_z"]["coef"] == pytest.approx(b_true, abs=0.08)


def test_overlay_macro_cols_preserve_inflationary(monkeypatch: pytest.MonkeyPatch) -> None:
    t, n = 12, 3
    returns = np.zeros((t, n))
    existing = np.array(
        ["calm", "inflationary", "calm"] + ["calm"] * 9, dtype=object
    )
    monkeypatch.setattr(
        "src.eval.turbulence.turbulence_index",
        lambda r, **kwargs: np.ones(t),
    )
    monkeypatch.setattr(
        "src.eval.turbulence.classify_regime",
        lambda turb, **kwargs: np.array([True, True, False] + [False] * 9),
    )
    macro = np.ones((t, 2))
    out = bm.turbulence_regimes_from_returns(
        returns,
        existing=existing,
        macro_cols=macro,
        overlay_mode="q75",
    )
    assert out[0] == "crisis"
    assert out[1] == "inflationary"
