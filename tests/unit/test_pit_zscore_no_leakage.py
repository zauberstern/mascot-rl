"""E-5: date-causal z-score / winsorize must not leak future rows."""
from __future__ import annotations

import numpy as np

from src.features.blocks.cross_section import apply_cross_section_normalize
from src.features.blocks.normalize import expanding_causal_zscore, normalize_cross_section_panel


def _truncation_invariant(fn, x: np.ndarray, *, min_obs: int = 2) -> None:
    t_len = x.shape[0]
    for t in range(t_len):
        if fn is expanding_causal_zscore:
            full_row = fn(x, min_obs=min_obs)[t]
            trunc_row = fn(x[: t + 1], min_obs=min_obs)[t]
        else:
            full_row = fn(x)[t]
            trunc_row = fn(x[: t + 1])[t]
        np.testing.assert_allclose(
            full_row,
            trunc_row,
            rtol=0,
            atol=1e-12,
            err_msg=f"leakage at t={t}",
            equal_nan=True,
        )


def test_expanding_causal_zscore_no_future_leakage() -> None:
    x = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [100.0, 200.0],
        ],
        dtype=np.float64,
    )
    _truncation_invariant(expanding_causal_zscore, x)


def test_cross_section_winsorize_zscore_no_future_leakage() -> None:
    x = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [100.0, 200.0],
        ],
        dtype=np.float64,
    )
    _truncation_invariant(normalize_cross_section_panel, x)


def test_apply_cross_section_normalize_per_channel_no_leakage() -> None:
    x = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [100.0, 200.0],
        ],
        dtype=np.float64,
    )
    cube = np.stack([x, x * 2.0], axis=-1)
    t_len = cube.shape[0]
    for t in range(t_len):
        full = apply_cross_section_normalize(cube)[t]
        trunc = apply_cross_section_normalize(cube[: t + 1])[t]
        np.testing.assert_allclose(full, trunc, rtol=0, atol=1e-12, equal_nan=True)
