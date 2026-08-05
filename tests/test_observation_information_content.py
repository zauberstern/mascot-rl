"""Observation channel matrix must carry >1 numerical rank (no sinusoid collapse)."""
from __future__ import annotations

import numpy as np

from mascotrl.features.raw_state import (
    build_raw_states,
    build_raw_states_from_feature_tensor,
    encode_scalar_series_legacy_sinusoid,
)


def _channel_matrix_rank(raw: np.ndarray, *, tol: float = 1e-5) -> int:
    """Rank of the (seq, d_model) matrix for a single asset (or stacked rows)."""
    assert raw.ndim == 2
    return int(np.linalg.matrix_rank(raw, tol=tol))


def test_legacy_sinusoid_rank_approx_one():
    """Constant ATM panels → raw tensors are scalar multiples of one template (rank-1)."""
    d_model = 16
    k, seq = 4, 24
    constants = np.linspace(0.1, 0.4, 12)
    # Flatten each encoding; rows differ only by the ATM scalar → rank 1.
    rows = []
    for c in constants:
        base = np.full((k, seq), float(c), dtype=np.float64)
        raw = encode_scalar_series_legacy_sinusoid(base, d_model=d_model)
        rows.append(raw.reshape(-1))
    mat = np.stack(rows, axis=0)
    assert _channel_matrix_rank(mat, tol=1e-4) == 1

    # Per observation, channels are a deterministic function of the scalar (DOF=1).
    base = np.full((k, seq), 0.25, dtype=np.float64)
    raw = encode_scalar_series_legacy_sinusoid(base, d_model=d_model)
    t_idx = np.arange(seq, dtype=np.float32)[None, :, None]
    k_idx = np.arange(k, dtype=np.float32)[:, None, None]
    recon = np.empty_like(raw)
    recon[:, :, 0] = base.astype(np.float32)
    for i in range(1, d_model):
        recon[:, :, i] = (
            base.astype(np.float32) * np.sin(0.1 * i * t_idx + 0.01 * i * k_idx)[..., 0]
        )
    np.testing.assert_allclose(raw, recon, atol=1e-6)


def test_multi_channel_feature_tensor_rank_gt_one():
    rng = np.random.default_rng(1)
    # Non-degenerate independent channels — per-asset (seq, d_model) rank > 1.
    feat = rng.normal(size=(4, 20, 6)).astype(np.float64)
    raw = build_raw_states_from_feature_tensor(feat, d_model=8)
    assert raw.shape == (4, 20, 8)
    for asset in range(4):
        assert _channel_matrix_rank(raw[asset], tol=1e-5) > 1


def test_build_raw_states_uses_feature_hist_not_sinusoid():
    rng = np.random.default_rng(2)
    k, seq, c, d_model = 3, 16, 5, 8
    atm = rng.normal(0.2, 0.01, size=(k, seq))
    feature_hist = rng.normal(size=(k, seq, c)).astype(np.float64)
    raw = build_raw_states(
        atm, d_model=d_model, feature_hist=feature_hist
    )
    legacy = encode_scalar_series_legacy_sinusoid(atm, d_model=d_model)
    assert raw.shape == (k, seq, d_model)
    assert _channel_matrix_rank(raw[0]) > 1
    assert not np.allclose(raw, legacy)


def test_legacy_encoder_flag_keeps_sinusoid():
    rng = np.random.default_rng(3)
    atm = rng.normal(0.2, 0.01, size=(2, 10))
    raw = build_raw_states(
        atm, d_model=8, feature_encoder="legacy_sinusoid"
    )
    expected = encode_scalar_series_legacy_sinusoid(atm, d_model=8)
    np.testing.assert_allclose(raw, expected)
    # Constant-panel rank-1 check on the same encoder path.
    rows = []
    for c in (0.15, 0.25, 0.35):
        r = build_raw_states(
            np.full_like(atm, c), d_model=8, feature_encoder="legacy_sinusoid"
        )
        rows.append(r.reshape(-1))
    assert _channel_matrix_rank(np.stack(rows), tol=1e-4) == 1


def test_pad_truncate_feature_tensor():
    feat_small = np.ones((2, 4, 3), dtype=np.float64)
    out = build_raw_states_from_feature_tensor(feat_small, d_model=5)
    assert out.shape == (2, 4, 5)
    np.testing.assert_allclose(out[..., :3], 1.0)
    np.testing.assert_allclose(out[..., 3:], 0.0)

    feat_big = np.arange(2 * 4 * 7, dtype=np.float64).reshape(2, 4, 7)
    out2 = build_raw_states_from_feature_tensor(feat_big, d_model=5)
    assert out2.shape == (2, 4, 5)
    np.testing.assert_allclose(out2, feat_big[..., :5].astype(np.float32))
