"""Liquidity proxies: Amihud illiquidity and dollar volume."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def map_wide_to_slots(
    wide: np.ndarray,
    *,
    secids: Sequence,
    slots_rows: Sequence[Sequence],
) -> np.ndarray:
    """Map fingerprint-wide ``(T, N)`` panel onto slotted ``(T, K)``.

    Under dynamic universes the ADV fingerprint often covers all historical
    occupants (N >> K). Day ``t`` column ``j`` receives the wide value of
    whichever secid occupies slot ``j`` that day (NaN if empty / unknown).
    """
    arr = np.asarray(wide, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"wide must be (T, N), got {arr.shape}")
    t_len, n = arr.shape
    if len(slots_rows) != t_len:
        raise ValueError(
            f"slots_rows length {len(slots_rows)} != wide T={t_len}"
        )
    if len(secids) != n:
        raise ValueError(f"secids length {len(secids)} != wide N={n}")
    col = {sid: i for i, sid in enumerate(secids)}
    k = len(slots_rows[0]) if slots_rows else 0
    out = np.full((t_len, k), np.nan, dtype=np.float64)
    for t, row in enumerate(slots_rows):
        if len(row) != k:
            raise ValueError(f"slots_rows[{t}] length {len(row)} != k={k}")
        for j, sid in enumerate(row):
            if sid is None:
                continue
            idx = col.get(sid)
            if idx is None:
                continue
            out[t, j] = arr[t, idx]
    return out


def amihud_illiquidity(
    returns: np.ndarray,
    dollar_volume: np.ndarray,
    *,
    window: int = 21,
) -> np.ndarray:
    """Amihud: mean(|r| / dollar_volume) over a causal window → ``(T, K)``."""
    from src.features.blocks.pandas_rolling import amihud_illiquidity_pandas

    return amihud_illiquidity_pandas(returns, dollar_volume, window=window)


def build_liquidity_block(
    returns: np.ndarray,
    *,
    dollar_volume: np.ndarray | None = None,
    window: int = 21,
    wide_dollar_volume: np.ndarray | None = None,
    wide_secids: Sequence | None = None,
    slots_rows: Sequence[Sequence] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Liquidity channels; empty cube if dollar_volume missing.

    Dynamic-universe path: pass fingerprint-wide ``wide_dollar_volume`` plus
    ``wide_secids`` / ``slots_rows`` to slot-align ADV onto ``(T, K)`` returns
    instead of dropping liquidity when N != K.
    """
    r = np.asarray(returns, dtype=np.float64)
    if dollar_volume is None and wide_dollar_volume is not None:
        if wide_secids is None or slots_rows is None:
            raise ValueError(
                "wide_dollar_volume requires wide_secids and slots_rows"
            )
        dollar_volume = map_wide_to_slots(
            wide_dollar_volume, secids=wide_secids, slots_rows=slots_rows
        )
    if dollar_volume is None:
        t, k = r.shape[:2]
        return np.zeros((t, k, 0), dtype=np.float64), []
    ami = amihud_illiquidity(r, dollar_volume, window=window)
    dv = np.asarray(dollar_volume, dtype=np.float64)
    # Trailing mean dollar volume as ADV proxy.
    t_len, k = dv.shape
    adv = np.full((t_len, k), np.nan, dtype=np.float64)
    w = int(window)
    for t in range(t_len):
        if t + 1 < w:
            continue
        block = dv[t + 1 - w : t + 1]
        adv[t] = np.nanmean(block, axis=0)
    cube = np.stack([ami, adv], axis=-1)
    return cube, ["amihud", "adv_dollar_volume"]
