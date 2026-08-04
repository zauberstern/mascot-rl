"""Mamba chunk clamp + shared-encoder vectorization equivalence."""
from __future__ import annotations

import torch

from src.features.extractor import AlphaFeatureExtractor
from src.features.mamba2 import PureTorchMamba2


def test_chunk_size_clamped_to_seqlen_matches_chunk1():
    torch.manual_seed(0)
    m32 = PureTorchMamba2(d_model=8, d_state=4, chunk_size=32)
    m1 = PureTorchMamba2(d_model=8, d_state=4, chunk_size=1)
    m1.load_state_dict(m32.state_dict())
    x = torch.randn(2, 1, 8)
    y32 = m32(x)
    y1 = m1(x)
    assert torch.allclose(y32, y1, atol=1e-5, rtol=1e-5)


def test_shared_encoder_vectorized_matches_loop_backends():
    torch.manual_seed(0)
    for backend in ("mamba", "gru", "lstm", "mlp", "transformer"):
        fe = AlphaFeatureExtractor(
            num_assets=5,
            d_model=8,
            d_state=4,
            temporal_backend=backend,
            use_dhgnn=False,
            share_temporal_encoder=True,
        )
        x = torch.randn(2, 5, 3, 8)
        # Reference: manual loop
        block = fe.temporal_blocks[0]
        outs = []
        for k in range(5):
            t = block(x[:, k, :, :])
            if t.dim() == 3:
                t = t[:, -1, :]
            outs.append(t)
        ref = torch.stack(outs, dim=1)
        got = fe(x, x[:, :, -1, :])
        assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5), backend
