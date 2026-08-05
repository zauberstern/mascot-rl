"""Align sealed operational Markov chronology onto a desk calendar.

Eval-only helper for Ch.10 Fixed-Share desk. Does not gate Fixed-Share alpha.
SCHEMA < 3 seals are refused (v2 turbulent was q75, not operational Markov).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from mascotrl.eval.regime_seal import load_sealed_series

MATCH_OK = 0.95
MATCH_PARTIAL = 0.80


def align_sealed_operational_mask(
    seal_path: Path | str,
    date_labels: Sequence[str],
) -> dict[str, Any]:
    """Load a SCHEMA>=3 seal and align operational columns to desk dates.

    Returns
    -------
    dict with status in {"ok","partial","unavailable"}, boolean/float arrays
    of length T=len(date_labels), matched_frac, and metadata.
    """
    path = Path(seal_path)
    t_len = len(date_labels)
    empty = {
        "status": "unavailable",
        "turbulent": None,
        "turbulent_q75": None,
        "turbulence": None,
        "hmm_p_highvol": None,
        "matched_frac": 0.0,
        "seal_name": path.name if path.name else None,
        "schema_version": None,
        "operational_label": None,
        "returns_source": None,
        "limitation": None,
    }
    if t_len == 0:
        empty["limitation"] = "empty desk date_labels"
        return empty
    if not path.is_dir() or not (path / "manifest.json").is_file():
        empty["limitation"] = f"missing seal directory: {path}"
        return empty

    try:
        frame, manifest = load_sealed_series(path)
    except Exception as exc:
        empty["limitation"] = f"load failed: {type(exc).__name__}: {exc}"
        return empty

    schema = int(manifest.get("schema_version") or 0)
    hp = manifest.get("hyperparams") or {}
    empty["schema_version"] = schema
    empty["seal_name"] = str(manifest.get("name") or path.name)
    empty["operational_label"] = hp.get("operational_label")
    empty["returns_source"] = hp.get("returns_source") or manifest.get(
        "returns_source"
    )

    if schema < 3:
        empty["limitation"] = (
            "seal schema v2; turbulent was q75; use usb_kpt10_v3"
        )
        empty["matched_frac"] = 0.0
        return empty

    # Index by ISO date string.
    if "date" in frame.columns and not isinstance(frame.index, pd.DatetimeIndex):
        idx = pd.to_datetime(frame["date"], errors="coerce")
    else:
        idx = pd.to_datetime(frame.index, errors="coerce")
    frame = frame.copy()
    frame["_iso"] = pd.DatetimeIndex(idx).strftime("%Y-%m-%d")
    by_date = frame.drop_duplicates("_iso", keep="last").set_index("_iso")

    turb = np.zeros(t_len, dtype=bool)
    q75 = np.zeros(t_len, dtype=bool)
    d_t = np.full(t_len, np.nan, dtype=np.float64)
    p_high = np.full(t_len, np.nan, dtype=np.float64)
    n_matched = 0
    for i, day in enumerate(date_labels):
        key = str(day)[:10]
        if key not in by_date.index:
            continue
        n_matched += 1
        row = by_date.loc[key]
        if "turbulent" in by_date.columns:
            turb[i] = bool(row["turbulent"])
        if "turbulent_q75" in by_date.columns:
            q75[i] = bool(row["turbulent_q75"])
        if "turbulence" in by_date.columns:
            d_t[i] = float(row["turbulence"])
        if "hmm_p_highvol" in by_date.columns:
            p_high[i] = float(row["hmm_p_highvol"])

    matched_frac = float(n_matched / t_len)
    limitation = None
    if matched_frac < MATCH_PARTIAL:
        status = "unavailable"
        limitation = (
            f"seal matched_frac={matched_frac:.3f} < {MATCH_PARTIAL}; "
            "fall back to live refit"
        )
        return {
            **empty,
            "status": status,
            "matched_frac": matched_frac,
            "limitation": limitation,
            "turbulent": None,
            "turbulent_q75": None,
            "turbulence": None,
            "hmm_p_highvol": None,
        }
    if matched_frac < MATCH_OK:
        status = "partial"
        limitation = f"seal matched_frac={matched_frac:.3f} < {MATCH_OK}"
    else:
        status = "ok"

    return {
        "status": status,
        "turbulent": turb,
        "turbulent_q75": q75,
        "turbulence": d_t,
        "hmm_p_highvol": p_high,
        "matched_frac": matched_frac,
        "seal_name": empty["seal_name"],
        "schema_version": schema,
        "operational_label": hp.get("operational_label") or "markov_filtered_p05",
        "returns_source": empty["returns_source"],
        "limitation": limitation,
    }
