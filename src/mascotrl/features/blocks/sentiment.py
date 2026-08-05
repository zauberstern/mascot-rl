"""Sentiment block: short interest + analyst recommendations / target gaps."""
from __future__ import annotations

from typing import Mapping

import numpy as np


def _as_tk(arr: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be (T,K), got {a.shape}")
    return a


def _lag_diff(panel: np.ndarray, lag: int) -> np.ndarray:
    out = np.full_like(panel, np.nan, dtype=np.float64)
    if lag <= 0:
        return out
    out[lag:] = panel[lag:] - panel[:-lag]
    return out


def build_sentiment_block(
    sentiment: Mapping[str, np.ndarray] | None,
) -> tuple[np.ndarray, list[str]]:
    if sentiment is None:
        return np.zeros((0, 0, 0), dtype=np.float64), []
    channels: list[np.ndarray] = []
    names: list[str] = []
    if "si_pct" in sentiment:
        si = _as_tk(sentiment["si_pct"], "si_pct")
        channels.append(si)
        names.append("si_pct")
        channels.append(_lag_diff(si, 21))
        names.append("si_pct_chg_21")
    if "rec_mean_inv" in sentiment:
        rec = _as_tk(sentiment["rec_mean_inv"], "rec_mean_inv")
        channels.append(rec)
        names.append("rec_mean_inv")
        channels.append(_lag_diff(rec, 63))
        names.append("rec_chg_63")
    if "pt_gap" in sentiment:
        channels.append(_as_tk(sentiment["pt_gap"], "pt_gap"))
        names.append("pt_gap")
    if not channels:
        return np.zeros((0, 0, 0), dtype=np.float64), []
    return np.stack(channels, axis=-1), names
