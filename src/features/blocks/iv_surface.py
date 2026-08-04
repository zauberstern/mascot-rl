"""IV surface / option-implied feature block for equity allocation obs."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

# Default channels when packing a signal dict / long panel into (T,K,C).
# Kept in sync with src.data.surface_signals.SURFACE_SIGNAL_NAMES (B4): every
# signal the gate can admit must also be a reachable observation channel.
DEFAULT_SURFACE_CHANNELS: tuple[str, ...] = (
    "iv_skew_30d",
    "iv_term_slope",
    "iv_convexity_30d",
    "cw_vol_spread",
    "vmp",
    "mfiv_30",
    "mfis_30",
    "mfik_30",
    "mfiv_365",
    "mfis_365",
    "mfik_365",
    "rns_term_spread",
    "svix2_30",
    "mw_xs",
    "d_iv_call_1m",
    "d_iv_put_1m",
    "surface_dispersion",
    "surface_quality",
    "os_ratio",
    "borrow_rate",
    "d_iv_term_slope_5d",
    "d_iv_skew_5d",
    "vrp_30",
)



def build_iv_surface_block(
    returns: np.ndarray,
    iv_surface: np.ndarray | Mapping[str, np.ndarray] | None = None,
    *,
    channel_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build ``(T, K, C)`` surface-signal block aligned to ``returns``.

    ``iv_surface`` may be:
    - ``(T, K, C)`` float array (already packed)
    - ``dict[str, (T, K)]`` named signal panels
    - ``None`` → empty block
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError(f"returns must be (T, K), got {r.shape}")
    t, k = r.shape
    if iv_surface is None:
        return np.zeros((t, k, 0), dtype=np.float64), []

    if isinstance(iv_surface, Mapping):
        names = list(channel_names) if channel_names is not None else [
            n for n in DEFAULT_SURFACE_CHANNELS if n in iv_surface
        ]
        if not names:
            names = [str(n) for n in iv_surface.keys()]
        cols = []
        used: list[str] = []
        for name in names:
            arr = iv_surface.get(name)
            if arr is None:
                continue
            a = np.asarray(arr, dtype=np.float64)
            if a.shape != (t, k):
                raise ValueError(
                    f"surface signal {name!r} shape {a.shape} != returns {(t, k)}"
                )
            cols.append(a)
            used.append(str(name))
        if not cols:
            return np.zeros((t, k, 0), dtype=np.float64), []
        cube = np.stack(cols, axis=-1)
        return cube, used

    cube = np.asarray(iv_surface, dtype=np.float64)
    if cube.ndim != 3 or cube.shape[0] != t or cube.shape[1] != k:
        raise ValueError(
            f"iv_surface array must be (T,K,C)=({t},{k},*) got {cube.shape}"
        )
    c = int(cube.shape[2])
    if channel_names is not None and len(channel_names) == c:
        names = [str(n) for n in channel_names]
    else:
        names = [f"surf_{i}" for i in range(c)]
    return cube, names


def build_borrow_block(
    returns: np.ndarray,
    borrow: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    r = np.asarray(returns, dtype=np.float64)
    t, k = r.shape
    if borrow is None:
        return np.zeros((t, k, 0), dtype=np.float64), []
    b = np.asarray(borrow, dtype=np.float64)
    if b.shape != (t, k):
        raise ValueError(f"borrow shape {b.shape} != {(t, k)}")
    return np.nan_to_num(b[..., None], nan=0.0), ["borrow_rate"]
