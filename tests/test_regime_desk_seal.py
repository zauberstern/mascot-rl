"""Align sealed operational Markov chronology onto desk date labels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval.regime_desk_seal import align_sealed_operational_mask


def _write_seal(
    dest: Path,
    *,
    dates: list[str],
    schema_version: int = 3,
    turbulent: np.ndarray | None = None,
    turbulent_q75: np.ndarray | None = None,
) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    t = len(dates)
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    if turbulent is None:
        turbulent = np.zeros(t, dtype=bool)
        turbulent[t // 2 :] = True
    if turbulent_q75 is None:
        turbulent_q75 = turbulent.copy()
        turbulent_q75[::3] = True
    frame = pd.DataFrame(
        {
            "turbulence": np.linspace(1.0, 5.0, t),
            "turbulent": turbulent.astype(bool),
            "turbulent_q75": turbulent_q75.astype(bool),
            "hmm_p_highvol": np.linspace(0.1, 0.9, t),
            "hmm_hard": turbulent.astype(np.int32),
            "regime": np.where(turbulent, "crisis", "calm"),
        },
        index=idx,
    )
    frame.to_parquet(dest / "regime_series.parquet")
    manifest = {
        "name": dest.name,
        "schema_version": schema_version,
        "hyperparams": {
            "operational_label": "markov_filtered_p05",
            "returns_source": "kpt10_gics",
        },
    }
    (dest / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dest


def test_align_perfect_match(tmp_path: Path) -> None:
    dates = pd.bdate_range("2020-01-01", periods=40).strftime("%Y-%m-%d").tolist()
    seal = _write_seal(tmp_path / "usb_kpt10_v3", dates=dates)
    out = align_sealed_operational_mask(seal, dates)
    assert out["status"] == "ok"
    assert out["matched_frac"] == pytest.approx(1.0)
    assert out["schema_version"] == 3
    assert out["operational_label"] == "markov_filtered_p05"
    assert len(out["turbulent"]) == 40
    assert bool(out["turbulent"][0]) is False
    assert bool(out["turbulent"][-1]) is True
    assert np.isfinite(out["turbulence"]).all()


def test_align_missing_middle_dates_partial_or_unavailable(tmp_path: Path) -> None:
    seal_dates = pd.bdate_range("2020-01-01", periods=20).strftime("%Y-%m-%d").tolist()
    seal = _write_seal(tmp_path / "seal", dates=seal_dates)
    # Desk calendar extends beyond seal and skips some days.
    desk = seal_dates[:10] + ["2020-06-01", "2020-06-02"] + seal_dates[10:]
    out = align_sealed_operational_mask(seal, desk)
    assert out["matched_frac"] < 1.0
    assert np.isnan(out["turbulence"][10])
    assert bool(out["turbulent"][10]) is False
    assert out["status"] in ("partial", "unavailable", "ok")


def test_align_schema_v2_unavailable_no_reinterpret(tmp_path: Path) -> None:
    dates = pd.bdate_range("2020-01-01", periods=30).strftime("%Y-%m-%d").tolist()
    seal = _write_seal(tmp_path / "usb_kpt10_v2", dates=dates, schema_version=2)
    out = align_sealed_operational_mask(seal, dates)
    assert out["status"] == "unavailable"
    assert "schema v2" in str(out["limitation"]).lower() or "v2" in str(
        out["limitation"]
    )
    # Must not treat v2 turbulent as operational Markov.
    assert out.get("turbulent") is None or out["matched_frac"] == 0.0


def test_align_low_match_unavailable(tmp_path: Path) -> None:
    seal_dates = pd.bdate_range("2010-01-01", periods=10).strftime("%Y-%m-%d").tolist()
    seal = _write_seal(tmp_path / "seal", dates=seal_dates)
    desk = pd.bdate_range("2020-01-01", periods=50).strftime("%Y-%m-%d").tolist()
    out = align_sealed_operational_mask(seal, desk)
    assert out["status"] == "unavailable"
    assert out["matched_frac"] < 0.80
