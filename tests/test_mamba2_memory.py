"""Memory guard for spectrum-grid PureTorchMamba2 on CPU (AWS m7i-flex.large)."""
from __future__ import annotations

import tracemalloc

import pytest
import torch

from src.features.mamba2 import PureTorchMamba2


def test_mamba2_spectrum_grid_peak_memory_cpu():
    """K=100 panel path: B*K flattened with d_model=64, seq_len=1, chunk_size=1."""
    d_model = 64
    k = 100
    seq_len = 1
    batch = 1
    # Match feature extractor flatten: (B*K, L, D)
    bsz = batch * k

    torch.manual_seed(42)
    model = PureTorchMamba2(
        d_model=d_model, d_state=16, d_conv=4, expand=2, chunk_size=1
    )
    x = torch.randn(bsz, seq_len, d_model, requires_grad=True)

    tracemalloc.start()
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert torch.isfinite(y).all()
    assert y.shape == (bsz, seq_len, d_model)
    # AWS job has ~8 GiB; keep Mamba forward+backward well under 500 MiB.
    assert peak < 500 * 1024 * 1024, f"peak={peak / (1024**2):.1f} MiB"
